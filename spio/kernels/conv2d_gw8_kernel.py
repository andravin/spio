"""Create the kernel factory for the Conv2d GW8 kernel."""

from dataclasses import dataclass
from itertools import product
from typing import Generator

from ..generators import *
from ..cuda.driver import DeviceAttributes

from ..util import divup, next_relative_prime
from .launch_params import LaunchParams
from .conv2d_gw8_params import Conv2dGw8Params
from .conv2d_stats import Conv2dStats
from .kernel_factory import KernelFactory, KernelSpec
from .kernel import get_full_kernel_name


@dataclass(frozen=True)
class Conv2dGw8Config:
    """Tile configuration for the Conv2d GW8 kernel."""

    groups: int = 8
    block_p: int = 16
    block_n: int = 1


def _get_configs(
    params: Conv2dGw8Params, _device_attr: DeviceAttributes, **_kwargs
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


def _get_kernel_spec(
    params: Conv2dGw8Params,
    config: Conv2dGw8Config,
    _device_attr: DeviceAttributes,
    igrad: bool = False,
) -> KernelSpec:
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
    blocks_c = divup(c, block_c)
    blocks = blocks_n * blocks_p * blocks_q * blocks_c
    warps = block_groups
    threads = warps * 32

    launch_params = LaunchParams(grid=blocks, block=threads)

    kernel_name = _get_kernel_name(igrad=igrad)
    full_kernel_name = get_full_kernel_name(kernel_name, params)

    kernel_has_bias = params.has_bias and not igrad

    # With 16 bytes-per-element, smem effectively has 8 banks.
    num_smem_banks = 8

    smem_x_stride = next_relative_prime(block_n * block_c8, num_smem_banks)

    gen_specs = [
        Macro({"SPIO_CONV_KERNEL": full_kernel_name}),
        Fold("block_n", "n", block_n),
        Fold("block_p", "p", block_p),
        Fold("block_q", "q", block_q),
        Fold("block_c", "c", block_c),
        ParamsSpec(
            "Block",
            {
                "threads": threads,
            },
        ),
        ParamsSpec(
            "Padding",
            {
                "h": padding_h,
                "w": padding_w,
            },
        ),
        ParamsSpec("Mode", {"igrad": igrad, "has_bias": kernel_has_bias}),
        Index(
            "BlockIdx",
            Dims(
                block_n=blocks_n,
                block_p=blocks_p,
                block_q=blocks_q,
                block_c=blocks_c,
            ),
        ),
        Index("InputIdx", Dims(n=block_n, x=block_w, c8=block_c8)),
        Tensor("Input", dtype.uint4, Dims(n=n, y=h, x=w, c8=c8), constant=True),
        Tensor("Bias", dtype.float2, Dims(k8=c8, k2=4), constant=True),
        Index("BiasIdx", Dims(k8=block_c8, lane=32)),
        Tensor("Output", dtype.uint4, Dims(n=n, p=p, q=q, k8=c8)),
        Tensor("Weights", dtype.uint4, Dims(k=c, r=r, s=s), constant=True),
        Tensor("SmemWeights", dtype.uint4, Dims(k=block_c, r=r, s=s)),
        Tensor(
            "ConstSmemWeights",
            dtype.uint4,
            Dims(k8=block_c8, k=8, r=r, s=s),
            constant=True,
        ),
        Index(
            "SmemWeightsLoadIdx",
            Dims(k8=block_c8, repeat=4, k=8),
            dummies=["repeat"],
        ),
        Tensor(
            "SmemInput",
            dtype.uint4,
            Dims(ping_pong=2, x=block_w, n=block_n, c8=block_c8),
            strides=Strides(x=smem_x_stride),
        ),
        Index(
            "SmemInputLoadIdx",
            Dims(
                c8=block_c8,
                repeat=32 // (block_q * block_n),
                x=block_q,
                n=block_n,
            ),
            dummies=["repeat"],
        ),
        Index("SmemOutputStoreIdx", Dims(k8=block_c8, lane=32)),
        Tensor(
            "SmemOutput",
            dtype.half2,
            Dims(q=block_q, n=block_n, k8=block_c8 + 1, k2=4),
        ),
        Tensor(
            "ConstSmemOutput",
            dtype.uint4,
            Dims(q=block_q, n=block_n, k8=block_c8 + 1),
            constant=True,
        ),
        Index("OutputStoreIdx", Dims(n=block_n, q=block_q, k8=block_c8)),
        Index("BlockQNIdx", Dims(q=block_q, n=block_n)),
        Fragment("Acc", FragmentType.M16_N8_F32_C, "qn", "k"),
        Fragment("In", FragmentType.M16_K8_F16_A, "qn", "c"),
        Fragment("Wgts", FragmentType.N8_K8_F16_B, "c", "k"),
        Tensor("WeightsReg", "Wgts", Dims(r=r, s=s)),
        Tensor("AccReg", "Acc", Dims(p=r)),
    ]
    return KernelSpec(gen_specs=gen_specs, launch_params=launch_params)


conv2d_gw8_kernel_factory = KernelFactory(
    Conv2dGw8Params,
    Conv2dGw8Config,
    Conv2dStats,
    kernel_name=_get_kernel_name,
    configs=_get_configs,
    kernel_spec=_get_kernel_spec,
    kernel_source_file="conv2d_gw8.cu",
    src_module="spio.src",
    includes_module="spio.include",
    perf_model_skip_params=["group_width", "stride"],
)
