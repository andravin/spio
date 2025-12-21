"""Test matrix multiplication using the checkerboard shared-memory layout."""

from dataclasses import dataclass


import torch

from spio.generators import *
from spio.compiler import compile_and_load_kernel
from spio.util import divup, assert_all_close_with_acc_depth, SixteenChannelsLast


@dataclass
class MmaConfig:
    """Configuration for the mma kernel."""

    warp_m: int = 32
    warp_n: int = 64
    chunk_k16: int = 2


def test_mma_checkerboard_16c_kernel():
    """Compile and run a GPU kernel that tests tensor core mma with checkerboard layout for smem."""

    m, n, k = (8192, 1024, 1024)

    # TODO: fix for warp_m=64, warp_n=32
    config = MmaConfig(warp_m=32, warp_n=64, chunk_k16=2)

    specs, grid, block, max_registers = _get_specs(m, n, k, config=config)

    A = torch.randn((m, k), dtype=torch.float16, device="cuda")
    B_trans = torch.randn((n, k), dtype=torch.float16, device="cuda")
    C = torch.zeros((m, n), dtype=torch.float16, device="cuda")

    A_16c = SixteenChannelsLast.format(A)
    B_16c = SixteenChannelsLast.format(B_trans)
    C_16c = SixteenChannelsLast.format(C)

    _, mma_kernel_16c = compile_and_load_kernel(
        kernel_name="mma_checkerboard_16c",
        source_file_name="mma_checkerboard_16c.cu",
        header_dict={"types.h": generate(specs)},
        src_module="spio.src_tests",
        lineinfo=True,
        max_registers=max_registers,
    )
    mma_kernel_16c.launch(grid, block, (C_16c, A_16c, B_16c))
    C = SixteenChannelsLast.unformat(C_16c)

    C_ref = torch.matmul(A, B_trans.transpose(0, 1))
    assert_all_close_with_acc_depth(C, C_ref, acc_depth=k)


def _get_specs(m: int, n: int, k: int, config: MmaConfig = None):
    """Return the generator specs, grid and block for the mma checkerboard kernel."""
    assert k % 16 == 0, "Kernel requires K to be a multiple of 16."
    assert n % 8 == 0, "Kernel requires N to be a multiple of 8."
    assert config.chunk_k16 in [1, 2], "Chunk size must be 1 or 2."

    block_x = 128

    warps_m = block_x // config.warp_m
    warps_n = block_x // config.warp_n
    warps = warps_m * warps_n
    threads = warps * 32

    blocks_m = divup(m, block_x)
    blocks_n = divup(n, block_x)

    grid = (blocks_n, blocks_m, 1)

    block = (threads, 1, 1)

    warp_m16 = config.warp_m // 16
    warp_n16 = config.warp_n // 16

    k16 = k // 16
    j16 = n // 16

    block_x8 = block_x // 8
    block_x16 = block_x // 16

    warp_n8 = config.warp_n // 8

    k_chunk = config.chunk_k16 * 16
    k_double_chunk = 2 * k_chunk

    # Use the Generators container for cleaner spec definitions
    g = Generators()

    # Fold dimensions
    g.BlockIdx = Coordinates(
        Fold("i", block_x, init=BuiltIn.BLOCK_IDX_Y),
        Fold("j", block_x, init=BuiltIn.BLOCK_IDX_X),
    )
    g.warp_i = Fold("i", config.warp_m)
    g.warp_j = Fold("j", config.warp_n)
    g.k_chunk = Fold("k", k_chunk)
    g.k_double_chunk = Fold("k", k_double_chunk)

    # Global memory tensors
    g.AGlobal = Tensor(dtype.uint4, Dims(k16=k16, i=m, k8=-1), constant=True)
    g.BGlobal = Tensor(dtype.uint4, Dims(k16=k16, j=n, k8=-1), constant=True)
    g.CGlobal = Tensor(dtype.uint4, Dims(j16=j16, i=m, j8=-1))

    # Shared memory tensors
    g.ASmem = Tensor(
        dtype.uint4,
        Dims(k_chunk=2, k16=config.chunk_k16, i16=block_x16, swizzle=32),
    )
    g.BSmem = Tensor(
        dtype.uint4,
        Dims(k_chunk=2, k16=config.chunk_k16, j16=block_x16, swizzle=32),
    )

    # Load indices
    g.ALoadGlobalIndex = CompoundIndex(
        Dims(i16=block_x16, i=-1, k8=2), init=BuiltIn.THREAD_IDX_X
    )
    g.BLoadGlobalIndex = CompoundIndex(
        Dims(j16=block_x16, j=-1, k8=2), init=BuiltIn.THREAD_IDX_X
    )
    g.ComputeIndex = CompoundIndex(
        Dims(warp_i=warps_m, warp_j=warps_n, lane=32), init=BuiltIn.THREAD_IDX_X
    )

    # Async loaders
    g.ALoader = AsyncStripLoader(
        smem_tensor=g.ASmem,
        gmem_tensor=g.AGlobal,
        minor_axis="k16",
        major_axis_size=block_x16,
        minor_axis_size=config.chunk_k16,
        num_warps=warps,
    )
    g.BLoader = AsyncStripLoader(
        smem_tensor=g.BSmem,
        gmem_tensor=g.BGlobal,
        minor_axis="k16",
        major_axis_size=block_x16,
        minor_axis_size=config.chunk_k16,
        num_warps=warps,
    )

    # MMA fragments
    g.AFragment = Fragment(FragmentType.M16_K16_F16_A, "i", "k")
    g.BFragment = Fragment(FragmentType.N16_K16_F16_B, "k", "j")
    g.CFragment = Fragment(FragmentType.M16_N16_F32_C, "i", "j")

    # Register tensors (use fragment type names as data types)
    g.AReg = Tensor(g.AFragment, Dims(k16=config.chunk_k16, i16=warp_m16))
    g.BReg = Tensor(g.BFragment, Dims(k16=config.chunk_k16, j16=warp_n16))
    g.CReg = Tensor(g.CFragment, Dims(i16=warp_m16, j16=warp_n16))

    # Matmul operation
    g.mma = Matmul(g.AReg, g.BReg, g.CReg, g.CReg, function_name="mma")

    # Swizzle patterns
    g.ASwizzle = Checkerboard(pairs_dim="i", colors_dim="k8", offset_dim="swizzle")
    g.BSwizzle = Checkerboard(pairs_dim="j", colors_dim="k8", offset_dim="swizzle")

    # Output transpose through shared memory
    g.CStoreSmem = Tensor(
        dtype.half2,
        Dims(warp_i=warps_m, j8=block_x8, i16=warp_m16, i=-1, j2=-1),
        strides=Strides(j8=(config.warp_m + 1) * 4),
    )
    g.CLoadSmem = Tensor(
        dtype.uint4,
        Dims(warp_i=warps_m, j8=block_x8, i=config.warp_m),
        strides=Strides(j8=(config.warp_m + 1)),
        constant=True,
    )
    g.CLoadSmemIndex = CompoundIndex(Dims(i=config.warp_m, j8=warp_n8))

    # Macro for loop unrolling
    g.macros = Macro(dict(MAIN_LOOP_UNROLL_DEPTH=""))

    if warps == 8:
        max_registers = 128
    elif warps == 4:
        max_registers = 255
    else:
        raise ValueError(f"Unsupported number of warps: {warps}")

    return (g, grid, block, max_registers)
