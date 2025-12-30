"""Test matrix multiplication using the checkerboard shared-memory layout."""

from dataclasses import dataclass

import pytest
import torch

from spio.generators import *
from spio.compiler import (
    compile_and_load_kernel,
    lineinfo,
    count_instructions,
    print_disasm,
)
from spio.util import divup, assert_all_close_with_acc_depth, SixteenChannelsLast


@dataclass
class MmaConfig:
    """Configuration for the mma kernel."""

    warp_m: int = 32
    warp_n: int = 64
    chunk_k16: int = 2


@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 1024, 1024),
        (8192 - 32, 1024, 1024 + 32),
    ],
)
def test_mma_checkerboard_16c_kernel(m, n, k):
    """Compile and run a GPU kernel that tests tensor core mma with checkerboard layout for smem."""

    # TODO: fix for warp_m=64, warp_n=32
    config = MmaConfig(warp_m=32, warp_n=64, chunk_k16=2)

    specs, grid, block, max_registers = _get_specs(m, n, k, config=config)

    A = torch.randn((m, k), dtype=torch.float16, device="cuda")
    B_trans = torch.randn((n, k), dtype=torch.float16, device="cuda")
    C = torch.zeros((m, n), dtype=torch.float16, device="cuda")

    A_16c = SixteenChannelsLast.format(A)
    B_16c = SixteenChannelsLast.format(B_trans)
    C_16c = SixteenChannelsLast.format(C)

    generated_code = generate(specs)

    _, mma_kernel_16c = compile_and_load_kernel(
        kernel_name="mma_checkerboard_16c",
        source_file_name="mma_checkerboard_16c.cu",
        header_dict={"types.h": generated_code},
        src_module="spio.src_tests",
        lineinfo=lineinfo.get(),
        max_registers=max_registers,
        count_instructions=count_instructions.get(),
        print_disasm=print_disasm.get(),
    )
    mma_kernel_16c.launch(grid, block, (C_16c, A_16c, B_16c))
    C = SixteenChannelsLast.unformat(C_16c)

    C_ref = torch.matmul(A, B_trans.transpose(0, 1))
    assert_all_close_with_acc_depth(C, C_ref, acc_depth=k)


def _get_specs(m: int, n: int, k: int, config: MmaConfig):
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

    # Thread-block coordinates.
    BlockIdx = Coordinates(
        Fold("i", block_x, init=BuiltIn.BLOCK_IDX_Y),
        Fold("j", block_x, init=BuiltIn.BLOCK_IDX_X),
    )

    # Fold dimensions
    g.warp_i = Fold("i", config.warp_m)
    g.warp_j = Fold("j", config.warp_n)
    g.k_chunk = Fold("k", k_chunk)
    g.k_double_chunk = Fold("k", k_double_chunk)

    # Load indices
    ALoadGlobalIndex = CompoundIndex(Dims(i=block_x, k8=2), init=BuiltIn.THREAD_IDX_X)
    BLoadGlobalIndex = CompoundIndex(Dims(j=block_x, k8=2), init=BuiltIn.THREAD_IDX_X)

    # The position of each warp's tile and the LANE dimension for each thread.
    LocalIndex = CompoundIndex(
        Dims(warp_i=warps_m, warp_j=warps_n, lane=32), init=BuiltIn.THREAD_IDX_X
    )

    # Global memory tensors
    g.AGlobal = Tensor(dtype.half8, Dims(k16=k16, i=m, k8=-1), constant=True)[
        ALoadGlobalIndex, BlockIdx
    ]
    g.BGlobal = Tensor(dtype.half8, Dims(k16=k16, j=n, k8=-1), constant=True)[
        BLoadGlobalIndex, BlockIdx
    ]
    g.CGlobal = Tensor(dtype.half8, Dims(j16=j16, i=m, j8=-1))[BlockIdx, LocalIndex]

    # Shared memory tensors
    g.ASmem = Tensor(
        dtype.half8,
        Dims(
            k_chunk=2,
            k16=config.chunk_k16,
            i16=block_x16,
            swizzle=Checkerboard(pairs_dim="i", colors_dim="k8", size=32),
        ),
    )
    g.BSmem = Tensor(
        dtype.half8,
        Dims(
            k_chunk=2,
            k16=config.chunk_k16,
            j16=block_x16,
            swizzle=Checkerboard(pairs_dim="j", colors_dim="k8", size=32),
        ),
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

    # Load and store views for the shared memory tensors.
    g.ALoadSmem = g.ASmem.with_dim(g.AFragment.load_index)[LocalIndex]
    g.BLoadSmem = g.BSmem.with_dim(g.BFragment.load_index)[LocalIndex]
    g.AStoreSmem = g.ASmem[ALoadGlobalIndex]
    g.BStoreSmem = g.BSmem[BLoadGlobalIndex]

    # Local memory tensors (i.e. registers)
    # - Each tensor "element" is itself a matrix fragment.
    g.AReg = Tensor(g.AFragment, Dims(k16=config.chunk_k16, i16=warp_m16))
    g.BReg = Tensor(g.BFragment, Dims(k16=config.chunk_k16, j16=warp_n16))
    g.CReg = Tensor(g.CFragment, Dims(i16=warp_m16, j16=warp_n16))

    # Matmul operation
    g.mma = Matmul(g.AReg, g.BReg, g.CReg, g.CReg)

    # Transpose output through shared memory
    g.CSmem = Tensor(
        dtype.half2,
        Dims(warp_i=warps_m, j8=block_x8, i=config.warp_m, j2=-1),
        strides=Strides(j8=(config.warp_m + 1) * 4),
    )
    g.CStoreSmem = g.CSmem.with_dim(g.CFragment.compound_index)[LocalIndex]
    g.CLoadSmem = g.CSmem.vector_length(8, constant=True)[LocalIndex]
    g.CLoadSmemIndex = CompoundIndex(Dims(i=config.warp_m, j8=warp_n8))

    g.c_output_idx = g.CLoadSmemIndex.partition("lane", LocalIndex)

    # Macro for loop unrolling
    g.macros = Macro(dict(MAIN_LOOP_UNROLL_DEPTH=""))

    if warps == 8:
        max_registers = 128
    elif warps == 4:
        max_registers = 255
    else:
        raise ValueError(f"Unsupported number of warps: {warps}")

    return (g, grid, block, max_registers)
