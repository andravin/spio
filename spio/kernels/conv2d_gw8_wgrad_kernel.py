"""Define the kernel factory for the Conv2dGw8Wgrad kernel."""

from dataclasses import dataclass
from itertools import product
from typing import Generator, Tuple, List

from ..generators import (
    GenSpecs,
    MacroSpec,
    ParamsSpec,
    IndexSpec,
    TensorSpec,
    FragmentSpec,
)
from ..util import divup
from .launch_params import LaunchParams
from .conv2d_gw8_params import Conv2dGw8Params
from .conv2d_stats import Conv2dStats
from .kernel_factory import make_kernel_factory
from .kernel import get_full_kernel_name


@dataclass(frozen=True)
class Conv2dGw8WgradConfig:
    """Tile configuration for the Conv2d GW8 Wgrad kernel."""

    groups: int = 8
    block_h: int = 16
    block_n_iters: int = 1
    warp_n: int = 1
    warp_s: int = None


KERNEL_NAME = "spio_conv2d_gw8_wgrad"

BLOCK_Q = 8


def _get_configs(
    params: Conv2dGw8Params,
) -> Generator[Conv2dGw8WgradConfig, None, None]:
    """Generate tile configurations for the given layer parameters."""
    max_groups = min(params.groups, 8)
    max_warps = 32

    s_up = divup(params.s, 2) * 2
    block_w = BLOCK_Q + s_up - 1

    # # Try configurations with warp_s = params.S.
    block_h_values = [
        block_h for block_h in [2, 4, 8, 16, 32, 64] if block_h <= params.h
    ]
    if params.h not in block_h_values:
        block_h_values.append(params.h)
    groups_values = [groups for groups in [2, 4, 8] if groups <= max_groups]
    if params.groups not in groups_values and params.groups <= max_groups:
        groups_values.append(params.groups)
    warp_s_values = [warp_s for warp_s in [1, 2] if warp_s <= params.s]
    if params.s not in warp_s_values:
        warp_s_values.append(params.s)
    block_n_iters_values = [
        block_n_iters
        for block_n_iters in [1, 2, 4, 8, 16, 32]
        if block_n_iters <= params.n
    ]
    warp_n_values = [warp_n for warp_n in [1, 2, 4] if warp_n <= params.n]
    yield from (
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
        if (groups * (divup(params.s, warp_s)) <= max_warps)
        # Ensure that a row of input values can be loaded with a single 128-bit load.
        and (warp_n * block_w <= 32 * divup(params.s, warp_s))
        # Avoid simulatenously large values of warp_n and groups.
        and (warp_n * groups <= max_groups * 2)
    )


def _get_specs(
    params: Conv2dGw8Params, config: Conv2dGw8WgradConfig = None, **_kwargs
) -> Tuple[List[GenSpecs], LaunchParams]:
    """Get the code gen specs and launch params."""
    params.validate()

    r, s = params.r, params.s

    n, c, h, w = params.n, params.c, params.h, params.w
    p, q = params.p, params.q
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
    block_groups = min(config.groups, groups)

    # Derived Tiles
    s_up = divup(s, 2) * 2
    block_c = block_groups * group_width
    block_c8 = block_c // 8
    block_w = BLOCK_Q + s_up - 1
    block_p = block_h + r - 1
    blocks_n = divup(n, block_n)
    blocks_h = divup(h, block_h)
    blocks_q = divup(q, BLOCK_Q)
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
            {"ping_pong": 2, "n": config.warp_n, "x": block_w, "c8": block_c8 + 1},
        ),
        TensorSpec(
            "SmemDelta",
            "uint4",
            {"ping_pong": 2, "n": config.warp_n, "q": BLOCK_Q, "k8": block_c8 + 1},
        ),
        TensorSpec("SmemWgrad", "float2", {"k8": block_c8, "s": s, "c": 8, "k2": 4}),
    ]

    # TODO: ensure that the smem tensors fit in the shared memory.
    # smem_size = smem_tensors[0].num_bytes + smem_tensors[1].num_bytes

    launch_params = LaunchParams(grid=blocks, block=threads)

    full_kernel_name = get_full_kernel_name(KERNEL_NAME, params)

    specs = [
        MacroSpec({"SPIO_CONV_WGRAD_KERNEL": full_kernel_name}),
        #
        # Block parameters.
        #
        ParamsSpec(
            "Block",
            {
                "n": block_n,
                "h": block_h,
                "q": BLOCK_Q,
                "c8": block_c8,
                "p": block_p,
                "threads": threads,
            },
        ),
        #
        # Constant parameters.
        #
        ParamsSpec(
            "Params",
            {
                "R": r,
                "S": s,
                "PADDING_H": padding_h,
                "PADDING_W": padding_w,
                "TRANSPOSE_PADDING_H": transpose_padding_h,
                "WARP_S": config.warp_s,
                "WARP_S2": warp_s2,
                "WARP_S2_UP": warp_s2_up,
                "BLOCK_N_ITERS": config.block_n_iters,
                "WARP_N": config.warp_n,
            },
        ),
        #
        # Block indices.
        #
        IndexSpec(
            "BlockIdx",
            {"n": blocks_n, "y": blocks_h, "q": blocks_q, "c8": blocks_c8},
        ),
        #
        # Input loading.
        #
        IndexSpec("InputIdx", {"n": config.warp_n, "x": block_w, "c8": block_c8}),
        TensorSpec("Input", "const uint4", {"n": n, "y": h, "x": w, "c8": c8}),
        IndexSpec(
            "SmemInputLoadIdx",
            {
                "c8": warps_c8,
                "warp_s": warps_s,
                "repeat": 32 // (2 * BLOCK_Q),
                "s": 2,
                "q": BLOCK_Q,
            },
        ),
        #
        # Delta loading
        #
        IndexSpec("DeltaIdx", {"n": config.warp_n, "q": BLOCK_Q, "k8": block_c8}),
        TensorSpec("Delta", "const uint4", {"n": n, "p": p, "q": q, "k8": c8}),
        IndexSpec(
            "SmemDeltaLoadIdx",
            {"k8": warps_c8, "repeat": (32 * warps_s) // BLOCK_Q, "q": BLOCK_Q},
        ),
        TensorSpec("DeltaFrag", "spio::MMA_N8_K8_F16_B", {"n": config.warp_n, "r": r}),
        #
        # Accumulator
        #
        FragmentSpec("Acc", "MMA_M16_N8_F32_C", "c", "k"),
        TensorSpec("AccTensor", "spio::MMA_M16_N8_F32_C", {"s2": warp_s2_up, "r": r}),
        #
        # Weights storing.
        #
        IndexSpec("SmemWgradStoreIdx", {"k8": warps_c8, "warp_s": warps_s, "lane": 32}),
        # Each thread stores 8k for a particular (k8, r, s, c).
        IndexSpec("WgradStoreIdx", {"k8": warps_c8, "s": s, "c": 8}),
        # Reduce Wgrad through global memory using float32 precision.
        TensorSpec("Wgrad", "float", {"k": c, "r": r, "s": s, "c": 8}),
    ] + smem_tensors
    return specs, launch_params


conv2d_gw8_wgrad_kernel_factory = make_kernel_factory(
    Conv2dGw8Params,
    Conv2dGw8WgradConfig,
    Conv2dStats,
    kernel_name=KERNEL_NAME,
    configs=_get_configs,
    specs=_get_specs,
    kernel_source_file="conv2d_gw8_wgrad.cu",
    src_module="spio.src",
    includes_module="spio.include",
    perf_model_skip_params=["group_width", "stride"],
)
