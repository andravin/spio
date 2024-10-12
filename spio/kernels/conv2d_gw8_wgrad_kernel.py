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

    Params = Conv2dGw8Params
    Config = Conv2dGw8WgradConfig
    Stats = Conv2dStats

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

        S_UP = divup(params.S, 2) * 2
        BLOCK_W = cls.BLOCK_Q + S_UP - 1

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
            and (warp_n * BLOCK_W <= 32 * divup(params.S, warp_s))
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

        R, S = params.R, params.S

        N, C, H, W = params.N, params.C, params.H, params.W
        P, Q = params.P, params.Q
        PADDING_H, PADDING_W = params.padding_h, params.padding_w
        TRANSPOSE_PADDING_H = params.transpose_padding_h

        # Hardcoded parameters:
        GROUP_WIDTH = params.group_width

        # Derived parameters
        C8 = C // 8
        GROUPS = params.groups

        # Tiles
        BLOCK_N = config.block_n_iters * config.warp_n
        BLOCK_H = min(config.block_h, H)
        BLOCK_Q = self.BLOCK_Q
        BLOCK_GROUPS = min(config.groups, GROUPS)

        # Derived Tiles
        S_UP = divup(S, 2) * 2
        BLOCK_C = BLOCK_GROUPS * GROUP_WIDTH
        BLOCK_C8 = BLOCK_C // 8
        BLOCK_W = BLOCK_Q + S_UP - 1
        BLOCK_P = BLOCK_H + R - 1
        BLOCKS_N = divup(N, BLOCK_N)
        BLOCKS_H = divup(H, BLOCK_H)
        BLOCKS_Q = divup(Q, BLOCK_Q)
        BLOCKS_C8 = divup(C8, BLOCK_C8)
        BLOCKS = BLOCKS_N * BLOCKS_H * BLOCKS_Q * BLOCKS_C8

        WARPS_C8 = BLOCK_C8
        WARPS_S = divup(S, config.warp_s)
        WARPS = WARPS_C8 * WARPS_S
        THREADS = WARPS * 32

        WARP_S2 = config.warp_s // 2
        WARP_S2_UP = divup(config.warp_s, 2)

        SMEM_TENSORS = [
            TensorSpec(
                "SmemInput",
                "uint4",
                dict(ping_pong=2, n=config.warp_n, x=BLOCK_W, c8=BLOCK_C8 + 1),
            ),
            TensorSpec(
                "SmemDelta",
                "uint4",
                dict(ping_pong=2, n=config.warp_n, q=BLOCK_Q, k8=BLOCK_C8 + 1),
            ),
            TensorSpec("SmemWgrad", "float2", dict(k8=BLOCK_C8, s=S, c=8, k2=4)),
        ]

        # TODO: ensure that the smem tensors fit in the shared memory.
        # smem_size = sum(tensor.num_bytes for tensor in SMEM_TENSORS)

        launch_params = LaunchParams(grid=BLOCKS, block=THREADS)

        kernel_name = self.get_kernel_name(params=params)

        specs = [
            MacroSpec(dict(SPIO_CONV_WGRAD_KERNEL=kernel_name)),
            #
            # Block parameters.
            #
            ParamsSpec(
                "Block",
                dict(
                    n=BLOCK_N,
                    h=BLOCK_H,
                    q=BLOCK_Q,
                    c8=BLOCK_C8,
                    p=BLOCK_P,
                    threads=THREADS,
                ),
            ),
            ParamsSpec(
                "Params",
                dict(
                    R=R,
                    S=S,
                    PADDING_H=PADDING_H,
                    PADDING_W=PADDING_W,
                    TRANSPOSE_PADDING_H=TRANSPOSE_PADDING_H,
                    WARP_S=config.warp_s,
                    WARP_S2=WARP_S2,
                    WARP_S2_UP=WARP_S2_UP,
                    BLOCK_N_ITERS=config.block_n_iters,
                    WARP_N=config.warp_n,
                ),
            ),
            IndexSpec(
                "BlockIdx",
                dict(n=BLOCKS_N, y=BLOCKS_H, q=BLOCKS_Q, c8=BLOCKS_C8),
            ),
            #
            # Input loading.
            #
            IndexSpec("InputIdx", dict(n=config.warp_n, x=BLOCK_W, c8=BLOCK_C8)),
            TensorSpec("Input", "const uint4", dict(n=N, y=H, x=W, c8=C8)),
            IndexSpec(
                "SmemInputLoadIdx",
                dict(
                    c8=WARPS_C8,
                    warp_s=WARPS_S,
                    repeat=32 // (2 * BLOCK_Q),
                    s=2,
                    q=BLOCK_Q,
                ),
            ),
            #
            # Delta loading
            #
            IndexSpec("DeltaIdx", dict(n=config.warp_n, q=BLOCK_Q, k8=BLOCK_C8)),
            TensorSpec("Delta", "const uint4", dict(n=N, p=P, q=Q, k8=C8)),
            IndexSpec(
                "SmemDeltaLoadIdx",
                dict(k8=WARPS_C8, repeat=(32 * WARPS_S) // BLOCK_Q, q=BLOCK_Q),
            ),
            TensorSpec(
                "DeltaFrag", "spio::MMA_N8_K8_F16_B", dict(n=config.warp_n, r=R)
            ),
            #
            # Accumulator
            #
            FragmentSpec("Acc", "MMA_M16_N8_F32_C", "c", "k"),
            TensorSpec("AccTensor", "spio::MMA_M16_N8_F32_C", dict(s2=WARP_S2_UP, r=R)),
            #
            # Weights storing.
            #
            IndexSpec("SmemWgradStoreIdx", dict(k8=WARPS_C8, warp_s=WARPS_S, lane=32)),
            # Each thread stores 8k for a particular (k8, r, s, c).
            IndexSpec("WgradStoreIdx", dict(k8=WARPS_C8, s=S, c=8)),
            # Reduce Wgrad through global memory using float32 precision.
            TensorSpec("Wgrad", "float", dict(k=C, r=R, s=S, c=8)),
        ] + SMEM_TENSORS

        super().__init__(
            self.get_kernel_name(params=params),
            launch_params,
            specs=specs,
            kernel_source_file="conv2d_gw8_wgrad.cu",
            params=params,
            config=config,
        )
