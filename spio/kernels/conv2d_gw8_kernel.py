from dataclasses import dataclass
from itertools import product
from math import prod

import torch

from ..generators import (
    generate,
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


@dataclass(frozen=True)
class Conv2dGw8Config:
    groups: int = 8
    block_p: int = 16
    block_n: int = 1


class Conv2dGw8Kernel(Kernel):

    Params = Conv2dGw8Params
    Config = Conv2dGw8Config

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
    def fprop_kernel(
        cls,
        params: Conv2dGw8Params,
        args,
        config: Conv2dGw8Config = None,
    ):
        return cls._fprop_kernel_cache.get(
            cls, params, args, config=config, igrad=False
        )

    @classmethod
    def grad_input_kernel(
        cls,
        params: Conv2dGw8Params,
        args,
        config: Conv2dGw8Config = None,
    ):
        return cls._dgrad_kernel_cache.get(cls, params, args, config=config, igrad=True)

    @classmethod
    def get_kernel(
        cls,
        params: Conv2dGw8Params,
        args,
        config: Conv2dGw8Config = None,
        igrad=False,
    ):
        if igrad:
            return cls.grad_input_kernel(params, args, config=config)
        else:
            return cls.fprop_kernel(params, args, config=config)

    def __init__(self, params, config=None, igrad=False):
        params.validate()

        R, S = params.R, params.S

        if igrad:
            N, C, H, W = params.N, params.C, params.P, params.Q
            P, Q = params.H, params.W
            PADDING_H, PADDING_W = (
                params.transpose_padding_h,
                params.transpose_padding_w,
            )
        else:
            N, C, H, W = params.N, params.C, params.H, params.W
            P, Q = params.P, params.Q
            PADDING_H, PADDING_W = params.padding_h, params.padding_w

        # Hardcoded parameter:
        GROUP_WIDTH = params.group_width

        # Derived parameters
        C8 = C // 8
        C2 = C // 2
        GROUPS = params.groups

        # Tiles
        BLOCK_N = min(config.block_n, N)
        BLOCK_P = min(config.block_p, P)
        BLOCK_Q = 16 // BLOCK_N
        BLOCK_GROUPS = min(config.groups, GROUPS)

        # Derived Tiles
        BLOCK_C = BLOCK_GROUPS * GROUP_WIDTH
        BLOCK_C8 = BLOCK_C // 8
        BLOCK_W = BLOCK_Q + S - 1
        BLOCKS_N = divup(N, BLOCK_N)
        BLOCKS_P = divup(P, BLOCK_P)
        BLOCKS_Q = divup(Q, BLOCK_Q)
        BLOCKS_C8 = divup(C8, BLOCK_C8)
        BLOCKS = BLOCKS_N * BLOCKS_P * BLOCKS_Q * BLOCKS_C8
        WARPS = BLOCK_GROUPS
        THREADS = WARPS * 32

        launch_params = LaunchParams(grid=BLOCKS, block=THREADS)

        kernel_name = "spio_conv2d_gw8_fprop" if not igrad else "spio_conv2d_gw8_dgrad"

        kernel_has_bias = params.has_bias and not igrad

        specs = [
            MacroSpec(dict(SPIO_CONV_KERNEL=kernel_name)),
            ParamsSpec(
                "Block",
                dict(
                    n=BLOCK_N,
                    p=BLOCK_P,
                    q=BLOCK_Q,
                    c8=BLOCK_C8,
                    padding_h=PADDING_H,
                    padding_w=PADDING_W,
                    threads=THREADS,
                ),
            ),
            ParamsSpec("Mode", dict(igrad=igrad, has_bias=kernel_has_bias)),
            IndexSpec(
                "BlockIdx",
                dict(n=BLOCKS_N, p=BLOCKS_P, q=BLOCKS_Q, c8=BLOCKS_C8),
            ),
            IndexSpec("InputIdx", dict(n=BLOCK_N, x=BLOCK_W, c8=BLOCK_C8)),
            TensorSpec("Input", "const uint4", dict(n=N, y=H, x=W, c8=C8)),
            TensorSpec("Bias", "const __half2", dict(k8=C8, k2=4)),
            IndexSpec("BiasIdx", dict(k8=BLOCK_C8, lane=32)),
            TensorSpec("Output", "uint4", dict(n=N, p=P, q=Q, k8=C8)),
            TensorSpec("Weights", "const uint4", dict(k=C, r=R, s=S)),
            TensorSpec("SmemWeights", "uint4", dict(k=BLOCK_C, r=R, s=S)),
            TensorSpec(
                "ConstSmemWeights",
                "const uint4",
                dict(kd8=BLOCK_C8, km8=8, rs=R * S),
            ),
            IndexSpec("SmemWeightsLoadIdx", dict(kd8=BLOCK_C8, rs=4, km8=8)),
            TensorSpec(
                "SmemInput",
                "uint4",
                dict(ping_pong=2, x=BLOCK_W, n=BLOCK_N, c8=BLOCK_C8 + 1),
            ),
            IndexSpec(
                "SmemInputLoadIdx",
                dict(
                    c8=BLOCK_C8,
                    repeat=32 // (BLOCK_Q * BLOCK_N),
                    q=BLOCK_Q,
                    n=BLOCK_N,
                ),
            ),
            IndexSpec("SmemOutputStoreIdx", dict(k8=BLOCK_C8, lane=32)),
            TensorSpec(
                "SmemOutput",
                "__half2",
                dict(q=BLOCK_Q, n=BLOCK_N, k8=BLOCK_C8 + 1, k2=4),
            ),
            TensorSpec(
                "ConstSmemOutput",
                "const uint4",
                dict(q=BLOCK_Q, n=BLOCK_N, k8=BLOCK_C8 + 1),
            ),
            IndexSpec("OutputStoreIdx", dict(n=BLOCK_N, q=BLOCK_Q, k8=BLOCK_C8)),
            FragmentSpec("Acc", "MMA_M16_N8_F32_C", "qn", "k"),
        ]

        super().__init__(
            kernel_name,
            launch_params,
            kernel_source_file="conv2d_gw8.cu",
            specs=specs,
            params=params,
            config=config,
        )

    @staticmethod
    def macs(params: Conv2dGw8Params, igrad=False):
        return prod(params.output_shape) * prod(
            (params.R, params.S, params.group_width)
        )

    @staticmethod
    def bytes_read(params: Conv2dGw8Params, igrad=False):
        input = prod(params.output_shape) if igrad else prod(params.input_shape)
        return (input + prod(params.weight_shape)) * 2

    @staticmethod
    def bytes_written(params: Conv2dGw8Params, igrad=False):
        output = prod(params.input_shape) if igrad else prod(params.output_shape)
        return output * 2
