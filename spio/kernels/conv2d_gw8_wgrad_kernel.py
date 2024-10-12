from dataclasses import dataclass
from itertools import product

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
class Conv2dGw8WgradConfig:
    groups: int = 8
    block_h: int = 16
    block_n_iters: int = 1
    warp_n: int = 1
    warp_s: int = None


class Conv2dGw8WgradKernel(Kernel):

    params_cls = Conv2dGw8Params
    config_cls = Conv2dGw8WgradConfig
    stats_cls = Conv2dStats

    _kernel_cache = KernelCache()

    BLOCK_Q = 8

    @classmethod
    def get_kernel_base_name(cls):
        return "spio_conv2d_gw8_wgrad"

    @classmethod
    def configs(cls, params: Conv2dGw8Params):
        configs = []

        max_groups = min(params.groups, 8)
        max_warps = 32

        s_up = divup(params.S, 2) * 2
        block_w = cls.BLOCK_Q + s_up - 1

        # # Try configurations with warp_s = params.S.
        block_h_values = [
            block_h for block_h in [2, 4, 8, 16, 32, 64] if block_h <= params.H
        ]
        if params.H not in block_h_values:
            block_h_values.append(params.H)
        groups_values = [groups for groups in [2, 4, 8] if groups <= max_groups]
        if params.groups not in groups_values and params.groups <= max_groups:
            groups_values.append(params.groups)
        warp_s_values = [warp_s for warp_s in [1, 2] if warp_s <= params.S]
        if params.S not in warp_s_values:
            warp_s_values.append(params.S)
        block_n_iters_values = [
            block_n_iters
            for block_n_iters in [1, 2, 4, 8, 16, 32]
            if block_n_iters <= params.N
        ]
        warp_n_values = [warp_n for warp_n in [1, 2, 4] if warp_n <= params.N]
        return [
            Conv2dGw8WgradConfig(
                groups=groups,
                block_h=block_h,
                warp_s=warp_s,
                block_n_iters=block_n_iters,
                warp_n=warp_n,
            )
            for groups, block_h, warp_s, block_n_iters, warp_n in product(
                groups_values,
                block_h_values,
                warp_s_values,
                block_n_iters_values,
                warp_n_values,
            )
            # Ensure that the number of groups does not exceed the hardware limit.
            if (groups * (divup(params.S, warp_s)) <= max_warps)
            # Ensure that a row of input values can be loaded with a single 128-bit load.
            and (warp_n * block_w <= 32 * divup(params.S, warp_s))
            # Avoid simulatenously large values of warp_n and groups.
            and (warp_n * groups <= max_groups * 2)
        ]

    @classmethod
    def get_kernel_cache(cls):
        return cls._kernel_cache

    @classmethod
    def grad_weight_kernel(cls, params: Conv2dGw8Params, device):
        return cls._kernel_cache.get(cls, params, device)

    def __init__(self, params, config=None):
        params.validate()

        r, s = params.R, params.S

        n, c, h, w = params.N, params.C, params.H, params.W
        p, q = params.P, params.Q
        padding_h, padding_w = params.padding_h, params.padding_w
        transpose_padding_h = params.transpose_padding_h

        # Hardcoded parameters:
        group_width = params.group_width

        # Derived parameters
        c8 = c // 8
        groups = params.groups

        # Tiles
        block_n = config.block_n_iters * config.warp_n
        block_h = min(config.block_h, h)
        block_q = self.BLOCK_Q
        block_groups = min(config.groups, groups)

        # Derived Tiles
        s_up = divup(s, 2) * 2
        block_c = block_groups * group_width
        block_c8 = block_c // 8
        block_w = block_q + s_up - 1
        block_p = block_h + r - 1
        blocks_n = divup(n, block_n)
        blocks_h = divup(h, block_h)
        blocks_q = divup(q, block_q)
        blocks_c8 = divup(c8, block_c8)
        blocks = blocks_n * blocks_h * blocks_q * blocks_c8

        warps_c8 = block_c8
        warps_s = divup(s, config.warp_s)
        warps = warps_c8 * warps_s
        threads = warps * 32

        warp_s2 = config.warp_s // 2
        warp_s2_up = divup(config.warp_s, 2)

        smem_tensors = [
            TensorSpec(
                "SmemInput",
                "uint4",
                dict(ping_pong=2, n=config.warp_n, x=block_w, c8=block_c8 + 1),
            ),
            TensorSpec(
                "SmemDelta",
                "uint4",
                dict(ping_pong=2, n=config.warp_n, q=block_q, k8=block_c8 + 1),
            ),
            TensorSpec("SmemWgrad", "float2", dict(k8=block_c8, s=s, c=8, k2=4)),
        ]

        # TODO: ensure that the smem tensors fit in the shared memory.
        # smem_size = smem_tensors[0].num_bytes + smem_tensors[1].num_bytes

        launch_params = LaunchParams(grid=blocks, block=threads)

        kernel_name = self.get_kernel_name(params=params)

        specs = [
            MacroSpec(dict(SPIO_CONV_WGRAD_KERNEL=kernel_name)),
            #
            # Block parameters.
            #
            ParamsSpec(
                "Block",
                dict(
                    n=block_n,
                    h=block_h,
                    q=block_q,
                    c8=block_c8,
                    p=block_p,
                    threads=threads,
                ),
            ),
            #
            # Constant parameters.
            #
            ParamsSpec(
                "Params",
                dict(
                    R=r,
                    S=s,
                    PADDING_H=padding_h,
                    PADDING_W=padding_w,
                    TRANSPOSE_PADDING_H=transpose_padding_h,
                    WARP_S=config.warp_s,
                    WARP_S2=warp_s2,
                    WARP_S2_UP=warp_s2_up,
                    BLOCK_N_ITERS=config.block_n_iters,
                    WARP_N=config.warp_n,
                ),
            ),
            #
            # Block indices.
            #
            IndexSpec(
                "BlockIdx",
                dict(n=blocks_n, y=blocks_h, q=blocks_q, c8=blocks_c8),
            ),
            #
            # Input loading.
            #
            IndexSpec("InputIdx", dict(n=config.warp_n, x=block_w, c8=block_c8)),
            TensorSpec("Input", "const uint4", dict(n=n, y=h, x=w, c8=c8)),
            IndexSpec(
                "SmemInputLoadIdx",
                dict(
                    c8=warps_c8,
                    warp_s=warps_s,
                    repeat=32 // (2 * block_q),
                    s=2,
                    q=block_q,
                ),
            ),
            #
            # Delta loading
            #
            IndexSpec("DeltaIdx", dict(n=config.warp_n, q=block_q, k8=block_c8)),
            TensorSpec("Delta", "const uint4", dict(n=n, p=p, q=q, k8=c8)),
            IndexSpec(
                "SmemDeltaLoadIdx",
                dict(k8=warps_c8, repeat=(32 * warps_s) // block_q, q=block_q),
            ),
            TensorSpec(
                "DeltaFrag", "spio::MMA_N8_K8_F16_B", dict(n=config.warp_n, r=r)
            ),
            #
            # Accumulator
            #
            FragmentSpec("Acc", "MMA_M16_N8_F32_C", "c", "k"),
            TensorSpec("AccTensor", "spio::MMA_M16_N8_F32_C", dict(s2=warp_s2_up, r=r)),
            #
            # Weights storing.
            #
            IndexSpec("SmemWgradStoreIdx", dict(k8=warps_c8, warp_s=warps_s, lane=32)),
            # Each thread stores 8k for a particular (k8, r, s, c).
            IndexSpec("WgradStoreIdx", dict(k8=warps_c8, s=s, c=8)),
            # Reduce Wgrad through global memory using float32 precision.
            TensorSpec("Wgrad", "float", dict(k=c, r=r, s=s, c=8)),
        ] + smem_tensors

        super().__init__(
            self.get_kernel_name(params=params),
            launch_params,
            specs=specs,
            kernel_source_file="conv2d_gw8_wgrad.cu",
            params=params,
            config=config,
        )
