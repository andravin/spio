"""Create the kernel factory for the Conv2d GW8 kernel."""

from dataclasses import dataclass
from itertools import product
from typing import Generator, Tuple, List

from ..generators import (
    MacroSpec,
    ParamsSpec,
    IndexSpec,
    TensorSpec,
    FragmentSpec,
    GenSpecs,
)
from ..util import divup
from .launch_params import LaunchParams
from .conv2d_gw8_params import Conv2dGw8Params
from .conv2d_stats import Conv2dStats
from .kernel_factory import make_kernel_factory
from .kernel import get_full_kernel_name


@dataclass(frozen=True)
class Conv2dGw8Config:
    """Tile configuration for the Conv2d GW8 kernel."""

    groups: int = 8
    block_p: int = 16
    block_n: int = 1


def _get_configs(
    params: Conv2dGw8Params, **_kwargs
) -> Generator[Conv2dGw8Config, None, None]:
    """Generate configurations for the Conv2d GW8 kernel."""
    # igrad is unused in this function
    max_groups = min(params.groups, 8)
    block_n_values = [block_n for block_n in [1, 2, 4] if block_n <= params.n]
    block_p_values = [
        block_p for block_p in [1, 2, 4, 8, 16, 32, 64] if block_p <= params.p
    ]
    if params.p not in block_p_values:
        block_p_values.append(params.p)
    groups_values = [groups for groups in [1, 2, 4, 8] if groups <= max_groups]
    if params.groups not in groups_values and params.groups <= max_groups:
        groups_values.append(params.groups)
    yield from (
        Conv2dGw8Config(groups=groups, block_p=block_p, block_n=block_n)
        for groups, block_p, block_n in product(
            groups_values, block_p_values, block_n_values
        )
    )


def _get_kernel_name(igrad=False) -> str:
    return "spio_conv2d_gw8_fprop" if not igrad else "spio_conv2d_gw8_dgrad"


def _get_specs(
    params: Conv2dGw8Params, config: Conv2dGw8Config = None, igrad: bool = False
) -> Tuple[List[GenSpecs], LaunchParams]:
    """The code generator specs and launch parameters."""
    params.validate()

    r, s = params.r, params.s

    if igrad:
        n, c, h, w = params.n, params.c, params.p, params.q
        p, q = params.h, params.w
        padding_h, padding_w = (
            params.transpose_padding_h,
            params.transpose_padding_w,
        )
    else:
        n, c, h, w = params.n, params.c, params.h, params.w
        p, q = params.p, params.q
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

    kernel_name = _get_kernel_name(igrad=igrad)
    full_kernel_name = get_full_kernel_name(kernel_name, params)

    kernel_has_bias = params.has_bias and not igrad

    specs = [
        MacroSpec({"SPIO_CONV_KERNEL": full_kernel_name}),
        ParamsSpec(
            "Block",
            {
                "n": block_n,
                "p": block_p,
                "q": block_q,
                "c8": block_c8,
                "padding_h": padding_h,
                "padding_w": padding_w,
                "threads": threads,
            },
        ),
        ParamsSpec("Mode", {"igrad": igrad, "has_bias": kernel_has_bias}),
        IndexSpec(
            "BlockIdx",
            {"n": blocks_n, "p": blocks_p, "q": blocks_q, "c8": blocks_c8},
        ),
        IndexSpec("InputIdx", {"n": block_n, "x": block_w, "c8": block_c8}),
        TensorSpec("Input", "const uint4", {"n": n, "y": h, "x": w, "c8": c8}),
        TensorSpec("Bias", "const float2", {"k8": c8, "k2": 4}),
        IndexSpec("BiasIdx", {"k8": block_c8, "lane": 32}),
        TensorSpec("Output", "uint4", {"n": n, "p": p, "q": q, "k8": c8}),
        TensorSpec("Weights", "const uint4", {"k": c, "r": r, "s": s}),
        TensorSpec("SmemWeights", "uint4", {"k": block_c, "r": r, "s": s}),
        TensorSpec(
            "ConstSmemWeights",
            "const uint4",
            {"kd8": block_c8, "km8": 8, "rs": r * s},
        ),
        IndexSpec("SmemWeightsLoadIdx", {"kd8": block_c8, "rs": 4, "km8": 8}),
        TensorSpec(
            "SmemInput",
            "uint4",
            {"ping_pong": 2, "x": block_w, "n": block_n, "c8": block_c8 + 1},
        ),
        IndexSpec(
            "SmemInputLoadIdx",
            {
                "c8": block_c8,
                "repeat": 32 // (block_q * block_n),
                "q": block_q,
                "n": block_n,
            },
        ),
        IndexSpec("SmemOutputStoreIdx", {"k8": block_c8, "lane": 32}),
        TensorSpec(
            "SmemOutput",
            "__half2",
            {"q": block_q, "n": block_n, "k8": block_c8 + 1, "k2": 4},
        ),
        TensorSpec(
            "ConstSmemOutput",
            "const uint4",
            {"q": block_q, "n": block_n, "k8": block_c8 + 1},
        ),
        IndexSpec("OutputStoreIdx", {"n": block_n, "q": block_q, "k8": block_c8}),
        IndexSpec("OutputQNIdx", {"q": block_q, "n": block_n}),
        FragmentSpec("Acc", "MMA_M16_N8_F32_C", "qn", "k"),
    ]
    return specs, launch_params


conv2d_gw8_kernel_factory = make_kernel_factory(
    Conv2dGw8Params,
    Conv2dGw8Config,
    Conv2dStats,
    kernel_name=_get_kernel_name,
    configs=_get_configs,
    specs=_get_specs,
    kernel_source_file="conv2d_gw8.cu",
    src_module="spio.src",
    includes_module="spio.include",
    perf_model_skip_params=["group_width", "stride"],
)
