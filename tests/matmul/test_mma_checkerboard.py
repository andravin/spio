"""Test matrix multiplication using the checkerboard shared-memory layout."""

from dataclasses import dataclass
from functools import partial

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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TOTAL_MATRIX_BYTES = 2 * 72 * 1024 * 1024


@dataclass
class MmaConfig:
    """Configuration for the mma kernel."""

    warp_m: int = 32
    warp_n: int = 64
    chunk_k16: int = 2


CONFIGS = [
    MmaConfig(warp_m=32, warp_n=64, chunk_k16=1),
    MmaConfig(warp_m=32, warp_n=64, chunk_k16=2),
    MmaConfig(warp_m=64, warp_n=32, chunk_k16=1),
    MmaConfig(warp_m=64, warp_n=32, chunk_k16=2),
    MmaConfig(warp_m=64, warp_n=64, chunk_k16=1),
    MmaConfig(warp_m=64, warp_n=64, chunk_k16=2),
]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 1024, 1024),
        (8192 - 32, 1024, 1024 + 32),
    ],
)
def test_mma_checkerboard_16c_kernel(m, n, k, config):
    """Compile and run a GPU kernel that tests tensor core mma with checkerboard layout for smem."""

    _, kernel, grid, block = _compile_kernel(m, n, k, config=config)

    A, B_trans, C = _make_matrices(m, n, k)
    A_16c, B_16c, C_16c = _to_sixteen_channels_last(A, B_trans, C)

    kernel.launch(grid, block, (C_16c, A_16c, B_16c))
    C_ref = torch.matmul(A, B_trans.transpose(0, 1))
    C = SixteenChannelsLast.unformat(C_16c)
    assert_all_close_with_acc_depth(C, C_ref, acc_depth=k)


def _get_specs(m: int, n: int, k: int, config: MmaConfig):
    """Return the generator specs, grid and block for the mma checkerboard kernel."""
    assert k % 16 == 0, "Kernel requires K to be a multiple of 16."
    assert n % 8 == 0, "Kernel requires N to be a multiple of 8."
    assert config.chunk_k16 in [1, 2], "Chunk size must be 1 or 2."

    # Calculate block and warp sizes.
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
    # The loader automatically chooses 1D or 2D based on num_warps vs inner_axis_size
    g.ALoader = AsyncStripLoader(
        smem_tensor=g.ASmem,
        gmem_tensor=g.AGlobal,
        inner_axis="i",
        outer_axis="k16",
        inner_axis_size=block_x16,
        outer_axis_size=config.chunk_k16,
        num_warps=warps,
    )
    g.BLoader = AsyncStripLoader(
        smem_tensor=g.BSmem,
        gmem_tensor=g.BGlobal,
        inner_axis="j",
        outer_axis="k16",
        inner_axis_size=block_x16,
        outer_axis_size=config.chunk_k16,
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _compile_kernel(m: int, n: int, k: int, config: MmaConfig):
    specs, grid, block, max_registers = _get_specs(m, n, k, config=config)

    generated_code = generate(specs)

    module, kernel = compile_and_load_kernel(
        kernel_name="mma_checkerboard_16c",
        source_file_name="mma_checkerboard_16c.cu",
        header_dict={"types.h": generated_code},
        src_module="spio.src_tests",
        lineinfo=lineinfo.get(),
        max_registers=max_registers,
        count_instructions=count_instructions.get(),
        print_disasm=print_disasm.get(),
    )
    return module, kernel, grid, block


def _make_matrices(m: int, n: int, k: int):
    A = torch.randn((m, k), dtype=torch.float16, device="cuda")
    B_trans = torch.randn((n, k), dtype=torch.float16, device="cuda")
    C = torch.zeros((m, n), dtype=torch.float16, device="cuda")
    return A, B_trans, C


def _to_sixteen_channels_last(*tensors):
    """Convert a list of tensors to SixteenChannelsLast format."""
    return tuple([SixteenChannelsLast.format(tensor) for tensor in tensors])


# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------


def _benchmark():
    """Benchmark all mma checkerboard configurations."""

    for config in CONFIGS:
        print()
        print(config)
        print(f"{'m':>6} {'n':>6} {'k':>6} {'ms':>6} {'TFLOP/s':>6}")
        for n in range(1024, 1024 * 9, 1024):
            _, kernel, grid, block = _compile_kernel(n, n, n, config=config)
            fn = partial(kernel.launch, grid, block)
            args_lists = _make_matrix_lists(n, n, n)
            _benchmark_kernel(fn, args_lists, n, n, n)

    print()
    print("PyTorch matmul for reference:")
    print(f"{'m':>6} {'n':>6} {'k':>6} {'ms':>6} {'TFLOP/s':>6}")
    for n in range(1 * 1024, 9 * 1024, 1024):
        args_lists = _make_torch_args_lists(n, n, n)
        _benchmark_kernel(torch.matmul, args_lists, n, n, n, args_not_tuple=True)


def _benchmark_kernel(
    fn,
    args_lists,
    m: int,
    n: int,
    k: int,
    warmup: int = 10,
    iters: int = 100,
    args_not_tuple: bool = False,
) -> None:
    """Benchmark a CUDA kernel function."""
    # Warmup
    for i in range(warmup):
        args_list = args_lists[i % len(args_lists)]
        if args_not_tuple:
            fn(*args_list)
        else:
            fn(args_list)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(iters):
        args_list = args_lists[(i + warmup) % len(args_lists)]
        if args_not_tuple:
            fn(*args_list)
        else:
            fn(args_list)
    end.record()
    end.synchronize()

    ms = start.elapsed_time(end) / iters

    flops = 2 * m * n * k  # for matmul
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"{m:6d} {n:6d} {k:6d} {ms:6.3f} {tflops:6.1f}")


def _make_matrix_lists(m, n, k, total_matrix_bytes=TOTAL_MATRIX_BYTES):
    args_lists = []
    num_matrices = _get_num_matrices(m, n, k, total_matrix_bytes)
    for _ in range(num_matrices):
        A, B_trans, C = _make_matrices(m, n, k)
        A_16c, B_16c, C_16c = _to_sixteen_channels_last(A, B_trans, C)
        args_lists.append((C_16c, A_16c, B_16c))
    return args_lists


def _make_torch_args_lists(m, n, k, total_matrix_bytes=TOTAL_MATRIX_BYTES):
    num_matrices = _get_num_matrices(m, n, k, total_matrix_bytes)
    args_lists = []
    for _ in range(num_matrices):
        A = torch.randn((m, k), dtype=torch.float16, device="cuda")
        B = torch.randn((k, n), dtype=torch.float16, device="cuda")
        args_lists.append((A, B))
    return args_lists


def _get_num_matrices(m, n, k, total_matrix_bytes=TOTAL_MATRIX_BYTES):
    matrix_bytes = (m * k + k * n + m * n) * 2
    num_matrices = divup(total_matrix_bytes, matrix_bytes)
    return num_matrices


if __name__ == "__main__":
    _benchmark()
