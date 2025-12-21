"""Define the kernel factory for the Conv2dGw8Wgrad kernel."""

from dataclasses import dataclass
from itertools import product
from typing import Generator

from ..generators import *
from ..util import divup, next_relative_prime
from ..cuda.driver import DeviceAttributes

from .launch_params import LaunchParams
from .conv2d_gw8_params import Conv2dGw8Params
from .conv2d_stats import Conv2dStats
from .kernel_factory import KernelFactory, KernelSpec
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
    _device_attr: DeviceAttributes,
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


def _get_kernel_spec(
    params: Conv2dGw8Params,
    config: Conv2dGw8WgradConfig,
    _device_attr: DeviceAttributes,
    **_kwargs,
) -> KernelSpec:
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

    # There are effectively 8 shared memory banks when storing 16-byte elements (8c float16).
    num_smem_banks = 8

    smem_x_stride = next_relative_prime(block_c8, num_smem_banks)

    smem_tensors = [
        Tensor(
            dtype.uint4,
            {"ping_pong": 2, "n": config.warp_n, "x": block_w, "c8": block_c8},
            class_name="SmemInput",
            strides={"x": smem_x_stride},
        ),
        Tensor(
            dtype.uint4,
            {"ping_pong": 2, "n": config.warp_n, "q": BLOCK_Q, "k8": block_c8},
            class_name="SmemDelta",
            strides={"q": smem_x_stride},
        ),
        Tensor(
            dtype.float2,
            {"k8": block_c8, "s": s, "c": 8, "k2": 4},
            class_name="SmemWgrad",
        ),
    ]

    smem_size = max(
        smem_tensors[0].num_bytes + smem_tensors[1].num_bytes, smem_tensors[2].num_bytes
    )
    assert smem_size < _device_attr.max_shared_memory_per_block

    launch_params = LaunchParams(grid=blocks, block=threads)

    full_kernel_name = get_full_kernel_name(KERNEL_NAME, params)

    gen_specs = [
        Macro({"SPIO_CONV_WGRAD_KERNEL": full_kernel_name}),
        #
        # Block parameters.
        #
        Fold("n", block_n, fold_name="block_n"),
        Fold("y", block_h, fold_name="block_y"),
        Fold("p", block_p, fold_name="block_p"),
        Fold("q", BLOCK_Q, fold_name="block_q"),
        Fold("c", block_c, fold_name="block_c"),
        Fold("s", config.warp_s, fold_name="warp_s"),
        ParamsSpec(
            "Block",
            {
                "threads": threads,
            },
        ),
        #
        # Constant parameters.
        #
        ParamsSpec(
            "Params",
            {
                "R_Size": r,
                "S_Size": s,
                "PADDING_H": padding_h,
                "PADDING_W": padding_w,
                "TRANSPOSE_PADDING_H": transpose_padding_h,
                "WARP_S_Size": config.warp_s,
                "WARP_S2_Size": warp_s2,
                "WARP_S2_UP_Size": warp_s2_up,
                "BLOCK_N_ITERS": config.block_n_iters,
                "WARP_N_Size": config.warp_n,
            },
        ),
        #
        # Block indices.
        #
        CompoundIndex(
            {
                "block_n": blocks_n,
                "block_y": blocks_h,
                "block_q": blocks_q,
                "block_c": blocks_c8,
            },
            class_name="BlockIdx",
        ),
        #
        # Input loading.
        #
        CompoundIndex(
            {"n": config.warp_n, "x": block_w, "c8": block_c8}, class_name="InputIdx"
        ),
        Tensor(
            dtype.uint4,
            {"n": n, "y": h, "x": w, "c8": c8},
            class_name="Input",
            constant=True,
        ),
        CompoundIndex(
            {
                "c8": warps_c8,
                "warp_s": warps_s,
                "repeat": 32 // (2 * BLOCK_Q),
                "s": 2,
                "q": BLOCK_Q,
            },
            class_name="SmemInputLoadIdx",
            dummies=["repeat"],
        ),
        #
        # Delta loading
        #
        CompoundIndex(
            {"n": config.warp_n, "q": BLOCK_Q, "k8": block_c8}, class_name="DeltaIdx"
        ),
        Tensor(
            dtype.uint4,
            {"n": n, "p": p, "q": q, "k8": c8},
            class_name="Delta",
            constant=True,
        ),
        CompoundIndex(
            {
                "k8": warps_c8,
                "repeat": (32 * warps_s) // BLOCK_Q,
                "q": BLOCK_Q,
            },
            class_name="SmemDeltaLoadIdx",
            dummies=["repeat"],
        ),
        #
        # MMA fragments
        #
        Fragment(FragmentType.M16_N8_F32_C, "c", "k", class_name="Acc"),
        Fragment(FragmentType.M16_K8_F16_A, "c", "x", class_name="InputFragment"),
        Fragment(FragmentType.N8_K8_F16_B, "x", "k", class_name="DeltaFragment"),
        #
        # MMA tensors
        #
        Tensor("Acc", {"s2": warp_s2_up, "r": r}, class_name="AccTensor"),
        Tensor("InputFragment", {"s2": warp_s2_up}, class_name="InputTensor"),
        Tensor("DeltaFragment", {"n": config.warp_n, "r": r}, class_name="DeltaTensor"),
        #
        # Weights storing.
        #
        CompoundIndex(
            {"k8": warps_c8, "warp_s": warps_s, "lane": 32},
            class_name="SmemWgradStoreIdx",
        ),
        # Each thread stores 8k for a particular (k8, r, s, c).
        CompoundIndex({"k8": warps_c8, "s": s, "c": 8}, class_name="WgradStoreIdx"),
        # Reduce Wgrad through global memory using float32 precision.
        Tensor(dtype.float, {"k": c, "r": r, "s": s, "c": 8}, class_name="Wgrad"),
    ] + smem_tensors
    return KernelSpec(gen_specs=gen_specs, launch_params=launch_params)


conv2d_gw8_wgrad_kernel_factory = KernelFactory(
    Conv2dGw8Params,
    Conv2dGw8WgradConfig,
    Conv2dStats,
    kernel_name=KERNEL_NAME,
    configs=_get_configs,
    kernel_spec=_get_kernel_spec,
    kernel_source_file="conv2d_gw8_wgrad.cu",
    src_module="spio.src",
    includes_module="spio.include",
    perf_model_skip_params=["group_width", "stride"],
)
