"""Create the kernel factory for the Conv2d GW8 kernel."""

from dataclasses import dataclass
from itertools import product
from typing import Generator

from ..generators import (
    CompoundIndex,
    Dim,
    dtype,
    Fragment,
    Generators,
    LANE,
    Macro,
    Tensor,
    Operand,
    ParamsSpec,
    BuiltIn,
)
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
    block_p_values = [block_p for block_p in [1, 2, 4, 8, 16, 32, 64] if block_p <= params.p]
    if params.p not in block_p_values:
        block_p_values.append(params.p)
    groups_values = [groups for groups in [1, 2, 4, 8] if groups <= max_groups]
    if params.groups not in groups_values and params.groups <= max_groups:
        groups_values.append(params.groups)
    yield from (
        Conv2dGw8Config(groups=groups, block_p=block_p, block_n=block_n)
        for groups, block_p, block_n in product(groups_values, block_p_values, block_n_values)
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
    groups = params.groups

    # Tiles
    block_n = min(config.block_n, n)
    block_p = min(config.block_p, p)
    block_q = 16 // block_n
    block_groups = min(config.groups, groups)

    # Derived Tiles
    block_c = block_groups * group_width
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

    smem_x_stride = next_relative_prime(block_n * (block_c // 8), num_smem_banks)

    g = Generators()

    g.macros = Macro({"SPIO_CONV_KERNEL": full_kernel_name})

    # Dimension types
    N = g.N = Dim()  # batch
    H = g.H = Dim()  # height (vertical spatial) - unified across input/output/filter
    W = g.W = Dim()  # width (horizontal spatial) - unified across input/output/filter
    C = g.C = Dim()  # input channels
    K = g.K = Dim()  # output channels

    # Sized dimension instances
    # Note: k and c have the same size for grouped conv, but k must be created
    # before c is reassigned from integer to StaticDim
    n = N(n)
    k = K(c)
    c = C(c)

    # Spatial extents - all use unified H and W
    # Note: p, q, r, s are reassigned from integers to StaticDims here
    y = H(h)  # input height
    x = W(w)  # input width
    p = H(p)  # output height
    q = W(q)  # output width
    r = H(r)  # filter height
    s = W(s)  # filter width

    lane = LANE(32)

    # Blocking folds (output-space blocking)
    BLOCK_N = g.BLOCK_N = g.N / block_n
    BLOCK_H = g.BLOCK_H = g.H / block_p
    BLOCK_W = g.BLOCK_W = g.W / block_q
    BLOCK_C = g.BLOCK_C = g.C / block_c
    BLOCK_K = g.BLOCK_K = g.K / block_c

    # Channel folds
    g.C8 = g.C / 8
    g.K8 = g.K / 8
    g.K2 = g.K / 2

    # Additional dimensions for shared memory and indices
    PING_PONG = g.PING_PONG = Dim()
    REPEAT = g.REPEAT = Dim()
    QN = g.QN = Dim()

    # Block-sized extents
    n_block = N(block_n)
    w_block = W(block_q)  # output width tile
    w_block_in = W(block_w)  # input width tile (includes filter halo)
    c_block = C(block_c)
    k_block = K(block_c)

    g.Block = ParamsSpec(dict(threads=threads))
    g.Padding = ParamsSpec(dict(h=padding_h, w=padding_w))
    g.Mode = ParamsSpec(dict(igrad=igrad, has_bias=kernel_has_bias))

    # Indices
    g.BlockIdx = CompoundIndex(
        (BLOCK_N(blocks_n), BLOCK_H(blocks_p), BLOCK_W(blocks_q), BLOCK_C(blocks_c)),
        init=BuiltIn.BLOCK_IDX_X,
    )
    OutputBlockIdx = CompoundIndex(
        (BLOCK_N(blocks_n), BLOCK_H(blocks_p), BLOCK_W(blocks_q), BLOCK_K(blocks_c)),
        init=BuiltIn.BLOCK_IDX_X,
    )
    g.InputIdx = CompoundIndex((n_block, w_block_in, c_block / 8), init=BuiltIn.THREAD_IDX_X)
    BiasIdx = CompoundIndex((k_block / 8, lane), init=BuiltIn.THREAD_IDX_X)
    g.OutputStoreIdx = CompoundIndex(n_block, w_block, k_block / 8, init=BuiltIn.THREAD_IDX_X)

    # MMA fragments
    g.Acc = Fragment(Operand.C, dtype.float, QN(16), K(8))
    g.In = Fragment(Operand.A, dtype.half, QN(16), C(8))
    g.Wgts = Fragment(Operand.B, dtype.half, C(8), K(8))

    # Tensors
    g.Input = Tensor((n, y, x, c / 8), dtype.uint4, constant=True)[
        (g.BlockIdx - W(padding_w)).drop(H), g.InputIdx
    ]
    g.Bias = Tensor((k / 8, (k / 2) % 4), dtype.float2, constant=True).with_dim(
        g.Acc.compound_index
    )[BiasIdx, OutputBlockIdx]
    g.Output = Tensor((n, p, q, k / 8), dtype.uint4)[OutputBlockIdx, g.OutputStoreIdx]
    g.Weights = Tensor((k, r, s), dtype.uint4, constant=True)
    g.SmemWeights = Tensor((k_block, r, s), dtype.uint4)

    # Smem double-buffering and thread repeat counts
    ping_pong = PING_PONG(2)
    repeat_weights = REPEAT(4)
    repeat_input = REPEAT(32 // (block_q * block_n))

    g.ConstSmemWeights = Tensor((k_block / 8, k_block % 8, r, s), dtype.uint4, constant=True)
    g.SmemWeightsLoadIdx = CompoundIndex(
        (k_block / 8, repeat_weights, k_block % 8), dummies=["repeat"], init=BuiltIn.THREAD_IDX_X
    )
    g.SmemWeightsLoad = g.ConstSmemWeights[g.SmemWeightsLoadIdx]
    g.SmemInput = Tensor(
        (ping_pong, w_block_in, n_block, c_block / 8),
        dtype.uint4,
        strides=W(smem_x_stride),
    )
    g.SmemInputStore = g.SmemInput[g.InputIdx]
    g.SmemInputLoadIdx = CompoundIndex(
        (c_block / 8, repeat_input, w_block, n_block),
        dummies=["repeat"],
        init=BuiltIn.THREAD_IDX_X,
    )
    g.SmemInputLoad = g.SmemInput[g.SmemInputLoadIdx]
    g.SmemOutputStoreIdx = CompoundIndex(k_block / 8, lane)
    g.SmemOutput = Tensor((w_block, n_block, (k_block / 8) + 1, (k_block / 2) % 4), dtype.half2)
    g.ConstSmemOutput = Tensor((w_block, n_block, (k_block / 8) + 1), dtype.uint4, constant=True)
    g.SmemOutputLoad = g.ConstSmemOutput[g.OutputStoreIdx]
    g.BlockQNIdx = CompoundIndex(w_block, n_block)

    # Register tensors of fragments
    g.WeightsReg = Tensor((r, s), g.Wgts)
    g.AccReg = Tensor((H(r.size),), g.Acc)

    return KernelSpec(gen_specs=list(g), launch_params=launch_params)


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
