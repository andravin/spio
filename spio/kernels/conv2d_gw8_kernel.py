from dataclasses import dataclass
from itertools import product
from math import prod

import torch

from ..generators import (
    MacroSpec,
    ParamsSpec,
    IndexSpec,
    TensorSpec,
    FragmentSpec,
)
from ..util import divup
from .kernel import Kernel
from .kernel_cache import KernelCache
from .launch_params import LaunchParams
from .conv2d_gw8_params import Conv2dGw8Params
from .conv2d_stats import Conv2dStats


@dataclass(frozen=True)
class Conv2dGw8Config:
    groups: int = 8
    block_p: int = 16
    block_n: int = 1


class Conv2dGw8Kernel(Kernel):

    params_cls = Conv2dGw8Params
    config_cls = Conv2dGw8Config
    stats_cls = Conv2dStats

    _fprop_kernel_cache = KernelCache()
    _dgrad_kernel_cache = KernelCache()

    @classmethod
    def configs(cls, params: Conv2dGw8Params):
        max_groups = min(params.groups, 8)
        block_n_values = [block_n for block_n in [1, 2, 4] if block_n <= params.N]
        block_p_values = [
            block_p for block_p in [1, 2, 4, 8, 16, 32, 64] if block_p <= params.P
        ]
        if params.P not in block_p_values:
            block_p_values.append(params.P)
        groups_values = [groups for groups in [1, 2, 4, 8] if groups <= max_groups]
        if params.groups not in groups_values and params.groups <= max_groups:
            groups_values.append(params.groups)
        yield from (
            Conv2dGw8Config(groups=groups, block_p=block_p, block_n=block_n)
            for groups, block_p, block_n in product(
                groups_values, block_p_values, block_n_values
            )
        )

    @classmethod
    def get_kernel_cache(cls, igrad=False):
        return cls._dgrad_kernel_cache if igrad else cls._fprop_kernel_cache

    @classmethod
    def fprop_kernel(cls, params: Conv2dGw8Params, device):
        return cls._fprop_kernel_cache.get(cls, params, device, igrad=False)

    @classmethod
    def grad_input_kernel(cls, params: Conv2dGw8Params, device):
        return cls._dgrad_kernel_cache.get(cls, params, device, igrad=True)

    @classmethod
    def get_kernel_base_name(cls, igrad=False) -> str:
        return "spio_conv2d_gw8_fprop" if not igrad else "spio_conv2d_gw8_dgrad"

    def __init__(self, params, config=None, igrad=False):
        params.validate()

        r, s = params.R, params.S

        if igrad:
            n, c, h, w = params.N, params.C, params.P, params.Q
            p, q = params.H, params.W
            padding_h, padding_w = (
                params.transpose_padding_h,
                params.transpose_padding_w,
            )
        else:
            n, c, h, w = params.N, params.C, params.H, params.W
            p, q = params.P, params.Q
            padding_h, padding_w = params.padding_h, params.padding_w

        # Hardcoded parameter:
        group_width = params.group_width

        # Derived parameters
        c8 = c // 8
        groups = params.groups

        # Tiles
        block_n = min(config.block_n, n)
        block_p = min(config.block_p, p)
        block_q = 16 // block_n
        block_groups = min(config.groups, groups)

        # Derived Tiles
        block_c = block_groups * group_width
        block_c8 = block_c // 8
        block_w = block_q + s - 1
        blocks_n = divup(n, block_n)
        blocks_p = divup(p, block_p)
        blocks_q = divup(q, block_q)
        blocks_c8 = divup(c8, block_c8)
        blocks = blocks_n * blocks_p * blocks_q * blocks_c8
        warps = block_groups
        threads = warps * 32

        launch_params = LaunchParams(grid=blocks, block=threads)

        kernel_name = self.get_kernel_name(params=params, igrad=igrad)

        kernel_has_bias = params.has_bias and not igrad

        specs = [
            MacroSpec(dict(SPIO_CONV_KERNEL=kernel_name)),
            ParamsSpec(
                "Block",
                dict(
                    n=block_n,
                    p=block_p,
                    q=block_q,
                    c8=block_c8,
                    padding_h=padding_h,
                    padding_w=padding_w,
                    threads=threads,
                ),
            ),
            ParamsSpec("Mode", dict(igrad=igrad, has_bias=kernel_has_bias)),
            IndexSpec(
                "BlockIdx",
                dict(n=blocks_n, p=blocks_p, q=blocks_q, c8=blocks_c8),
            ),
            IndexSpec("InputIdx", dict(n=block_n, x=block_w, c8=block_c8)),
            TensorSpec("Input", "const uint4", dict(n=n, y=h, x=w, c8=c8)),
            TensorSpec("Bias", "const __half2", dict(k8=c8, k2=4)),
            IndexSpec("BiasIdx", dict(k8=block_c8, lane=32)),
            TensorSpec("Output", "uint4", dict(n=n, p=p, q=q, k8=c8)),
            TensorSpec("Weights", "const uint4", dict(k=c, r=r, s=s)),
            TensorSpec("SmemWeights", "uint4", dict(k=block_c, r=r, s=s)),
            TensorSpec(
                "ConstSmemWeights",
                "const uint4",
                dict(kd8=block_c8, km8=8, rs=r * s),
            ),
            IndexSpec("SmemWeightsLoadIdx", dict(kd8=block_c8, rs=4, km8=8)),
            TensorSpec(
                "SmemInput",
                "uint4",
                dict(ping_pong=2, x=block_w, n=block_n, c8=block_c8 + 1),
            ),
            IndexSpec(
                "SmemInputLoadIdx",
                dict(
                    c8=block_c8,
                    repeat=32 // (block_q * block_n),
                    q=block_q,
                    n=block_n,
                ),
            ),
            IndexSpec("SmemOutputStoreIdx", dict(k8=block_c8, lane=32)),
            TensorSpec(
                "SmemOutput",
                "__half2",
                dict(q=block_q, n=block_n, k8=block_c8 + 1, k2=4),
            ),
            TensorSpec(
                "ConstSmemOutput",
                "const uint4",
                dict(q=block_q, n=block_n, k8=block_c8 + 1),
            ),
            IndexSpec("OutputStoreIdx", dict(n=block_n, q=block_q, k8=block_c8)),
            FragmentSpec("Acc", "MMA_M16_N8_F32_C", "qn", "k"),
        ]

        self.igrad = igrad

        super().__init__(
            kernel_name,
            launch_params,
            kernel_source_file="conv2d_gw8.cu",
            specs=specs,
            params=params,
            config=config,
        )
