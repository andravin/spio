"""Test matrix multiplication using the checkerboard shared-memory layout."""

from dataclasses import dataclass, replace
from functools import partial
from itertools import product
import argparse

import pytest
import torch

from spio.generators import *
from spio.compiler import (
    compile_and_load_kernel,
    lineinfo,
    count_instructions,
    print_disasm,
)
from spio.util import divup, assert_all_close_with_acc_depth, TwoFold
from spio.cuda import driver


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Size this greater than the L2 cache size to ensure cold cache during benchmarking.
TOTAL_MATRIX_BYTES = 2 * 72 * 1024 * 1024


@dataclass
class MmaConfig:
    """Configuration for the mma kernel."""

    warp_m: int = 32
    warp_n: int = 64
    warps_m: int = 4
    warps_n: int = 2
    chunk_k16: int = 2
    wave_size: int = 4

    unroll_depth: int = 1

    @property
    def unroll_depth_str(self) -> str:
        """Return a string representation of the unroll depth."""
        if self.unroll_depth is None:
            return ""
        return str(self.unroll_depth)

    @property
    def block_m(self) -> int:
        """Return the block M size."""
        return self.warp_m * self.warps_m

    @property
    def block_n(self) -> int:
        """Return the block N size."""
        return self.warp_n * self.warps_n

    @property
    def warps(self) -> int:
        """Return the total number of warps."""
        return self.warps_m * self.warps_n


WAVE_SIZE = [2, 4, 8]

BASE_CONFIGS = [
    MmaConfig(warp_m=64, warp_n=32, warps_m=2, warps_n=4, chunk_k16=2),
    MmaConfig(warp_m=64, warp_n=32, warps_m=2, warps_n=4, chunk_k16=4),
    MmaConfig(warp_m=64, warp_n=64, warps_m=2, warps_n=4, chunk_k16=2),
    MmaConfig(warp_m=64, warp_n=64, warps_m=2, warps_n=4, chunk_k16=4),
]

CONFIGS = [
    replace(config, wave_size=wave_size)
    for config, wave_size in product(BASE_CONFIGS, WAVE_SIZE)
]


class ConfigError(Exception):
    """Raised when a kernel configuration is invalid."""


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 1024, 1024),
    ],
)
def test_mma_checkerboard_16c_kernel(m: int, n: int, k: int, config: MmaConfig):
    """Compile and run a GPU kernel that tests tensor core mma with checkerboard layout for smem."""

    try:
        _, kernel, grid, block, smem_size = _compile_kernel(m, n, k, config=config)
    except ConfigError as e:
        pytest.skip(str(e))
        return

    A, B_trans, C = _make_matrices(m, n, k)
    A_16c, B_16c, C_16c = _to_two_fold(A, B_trans, C)

    kernel.launch(grid, block, (C_16c, A_16c, B_16c), shared_mem_bytes=smem_size)
    C_ref = torch.matmul(A, B_trans.transpose(0, 1))
    C = TwoFold(16, 16).unformat(C_16c)
    assert_all_close_with_acc_depth(C, C_ref, acc_depth=k)


def _get_specs(m: int, n: int, k: int, config: MmaConfig):
    """Return the generator specs, grid and block for the mma checkerboard kernel."""
    assert k % 128 == 0, "Kernel requires K to be a multiple of 128."
    assert n % 128 == 0, "Kernel requires N to be a multiple of 128."
    assert m % 128 == 0, "Kernel requires M to be a multiple of 128."

    k_chunk = config.chunk_k16 * 16
    if k % k_chunk != 0:
        raise ConfigError(
            f"K ({k}) must be a multiple of k_chunk ({k_chunk}) for this config."
        )

    # Calculate block and warp sizes.

    warps_m = config.warps_m
    warps_n = config.warps_n
    threads = config.warps * 32

    blocks_m = divup(m, config.block_m)
    blocks_n = divup(n, config.block_n)
    blocks = blocks_m * blocks_n
    grid = (blocks, 1, 1)

    block = (threads, 1, 1)

    i16 = m // 16
    k16 = k // 16
    j16 = n // 16

    block_m16 = config.block_m // 16
    block_n8 = config.block_n // 8
    block_n16 = config.block_n // 16

    warp_m16 = config.warp_m // 16
    warp_n16 = config.warp_n // 16
    warp_n8 = config.warp_n // 8

    k_chunk = config.chunk_k16 * 16
    k_double_chunk = 2 * k_chunk

    # Use the Generators container for cleaner spec definitions
    g = Generators()

    # Thread-block coordinates.
    block_waves = blocks_m // config.wave_size
    block_local = config.wave_size
    wave_stride = config.wave_size * config.block_m

    g.block_i_wave = Fold("i", wave_stride)
    g.block_i_local = Fold("i", config.block_m)
    g.block_j = Fold("j", config.block_n)

    BlockIdx = CompoundIndex(
        Dims(block_i_wave=block_waves, block_j=blocks_n, block_i_local=block_local),
        init=BuiltIn.BLOCK_IDX_X,
    )

    # Fold dimensions
    g.warp_i = Fold("i", config.warp_m)
    g.warp_j = Fold("j", config.warp_n)
    g.k_chunk = Fold("k", k_chunk)
    g.k_double_chunk = Fold("k", k_double_chunk)

    # Load indices
    ALoadGlobalIndex = CompoundIndex(
        Dims(i=config.block_m, k8=2), init=BuiltIn.THREAD_IDX_X
    )
    BLoadGlobalIndex = CompoundIndex(
        Dims(j=config.block_n, k8=2), init=BuiltIn.THREAD_IDX_X
    )

    # The position of each warp's tile and the LANE dimension for each thread.
    LocalIndex = CompoundIndex(
        Dims(warp_i=warps_m, warp_j=warps_n, lane=32), init=BuiltIn.THREAD_IDX_X
    )

    # Global memory tensors
    g.AGlobal = Tensor(dtype.half8, Dims(i16=i16, k16=k16, i=-1, k8=-1), constant=True)[
        ALoadGlobalIndex, BlockIdx
    ]
    g.BGlobal = Tensor(dtype.half8, Dims(j16=j16, k16=k16, j=-1, k8=-1), constant=True)[
        BLoadGlobalIndex, BlockIdx
    ]
    g.CGlobal = Tensor(dtype.half8, Dims(i16=i16, j16=j16, i=-1, j8=-1))[
        BlockIdx, LocalIndex
    ]

    # Shared memory tensors
    g.ASmem = Tensor(
        dtype.half8,
        Dims(
            k16=2 * config.chunk_k16,
            i16=block_m16,
            swizzle=Checkerboard(pairs_dim="i", colors_dim="k8", size=32),
        ),
    )
    g.BSmem = Tensor(
        dtype.half8,
        Dims(
            k16=2 * config.chunk_k16,
            j16=block_n16,
            swizzle=Checkerboard(pairs_dim="j", colors_dim="k8", size=32),
        ),
    )
    g.CSmem = Tensor(
        dtype.half2,
        Dims(warp_i=warps_m, j8=block_n8, i=config.warp_m, j2=-1),
        strides=Strides(j8=(config.warp_m + 1) * 4),
    )

    smem_size = max(g.ASmem.num_bytes + g.BSmem.num_bytes, g.CSmem.num_bytes)
    smem_capacity = driver.get_max_shared_memory_per_block_optin()

    if smem_size > smem_capacity:
        raise ConfigError(
            blocks_m
            % f"Required shared memory size {smem_size} exceeds device capacity {smem_capacity}"
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

    # Async loaders
    # The loader automatically chooses 1D or 2D based on num_warps vs inner_axis_size
    g.ALoader = AsyncStripLoader(
        smem_tensor=g.AStoreSmem,
        gmem_tensor=g.AGlobal,
        inner_axis="i16",
        outer_axis="k16",
        inner_axis_size=block_m16,
        outer_axis_size=config.chunk_k16,
        num_warps=config.warps,
        num_buffers=2,
    )
    g.BLoader = AsyncStripLoader(
        smem_tensor=g.BStoreSmem,
        gmem_tensor=g.BGlobal,
        inner_axis="j16",
        outer_axis="k16",
        inner_axis_size=block_n16,
        outer_axis_size=config.chunk_k16,
        num_warps=config.warps,
        num_buffers=2,
    )

    # Local memory tensors (i.e. registers)
    # - Each tensor "element" is itself a matrix fragment.
    g.AReg = Tensor(g.AFragment, Dims(k16=config.chunk_k16, i16=warp_m16))
    g.BReg = Tensor(g.BFragment, Dims(k16=config.chunk_k16, j16=warp_n16))
    g.CReg = Tensor(g.CFragment, Dims(i16=warp_m16, j16=warp_n16))

    # Matmul operation
    g.mma = Matmul(g.AReg, g.BReg, g.CReg, g.CReg)

    # Transpose output through shared memory
    g.CStoreSmem = g.CSmem.with_dim(g.CFragment.compound_index)[LocalIndex]
    g.CLoadSmem = g.CSmem.vector_length(8, constant=True)[LocalIndex]
    g.CLoadSmemIndex = CompoundIndex(Dims(i=config.warp_m, j8=warp_n8))

    g.c_output_idx = g.CLoadSmemIndex.partition("lane", LocalIndex)

    # Macro for loop unrolling
    g.macros = Macro(dict(MAIN_LOOP_UNROLL_DEPTH=config.unroll_depth_str))

    num_warp_elements = config.warp_m * config.warp_n
    if num_warp_elements > 64 * 32:
        max_registers = 255
    else:
        max_registers = 128

    return (g, grid, block, max_registers, smem_size)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _compile_kernel(m: int, n: int, k: int, config: MmaConfig):
    specs, grid, block, max_registers, smem_size = _get_specs(m, n, k, config=config)

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
    kernel.set_max_dynamic_shared_memory_size(smem_size)
    kernel.set_preferred_shared_memory_carveout(percentage=100)
    return module, kernel, grid, block, smem_size


def _make_matrices(m: int, n: int, k: int):
    A = torch.randn((m, k), dtype=torch.float16, device="cuda")
    B_trans = torch.randn((n, k), dtype=torch.float16, device="cuda")
    C = torch.zeros((m, n), dtype=torch.float16, device="cuda")
    return A, B_trans, C


def _to_two_fold(*tensors):
    """Convert a list of tensors to TwoFold format."""
    return tuple([TwoFold(16, 16).format(tensor) for tensor in tensors])


# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------


def _print_header():
    print(f"{'m':>6} {'n':>6} {'k':>6} {'ms':>6} {'TFLOP/s':>6}")


def _benchmark():
    """Benchmark all mma checkerboard configurations."""

    for config in CONFIGS:
        print()
        print(config)
        _print_header()
        for n in range(1024, 1024 * 9, 1024):
            try:
                _, kernel, grid, block, smem_size = _compile_kernel(
                    n, n, n, config=config
                )
                fn = partial(kernel.launch, grid, block, shared_mem_bytes=smem_size)
                args_lists = _make_matrix_lists(n, n, n)
                _benchmark_kernel(fn, args_lists, n, n, n)
            except ConfigError:
                pass

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
        A_16c, B_16c, C_16c = _to_two_fold(A, B_trans, C)
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


def _benchmark_single(
    m: int,
    n: int,
    k: int,
    config: MmaConfig,
    warmup: int = 10,
    iters: int = 100,
    pytorch: bool = False,
):
    """Benchmark a single kernel configuration against PyTorch."""
    print(f"Benchmarking: m={m}, n={n}, k={k}")
    print(
        f"Config: warp_m={config.warp_m}, warp_n={config.warp_n}, "
        f"chunk_k16={config.chunk_k16}, unroll_depth={config.unroll_depth}"
    )
    print(f"Warmup: {warmup}, Iterations: {iters}")
    print()

    # Benchmark spio kernel
    try:
        _, kernel, grid, block, smem_size = _compile_kernel(m, n, k, config=config)
        fn = partial(kernel.launch, grid, block, shared_mem_bytes=smem_size)
        args_lists = _make_matrix_lists(m, n, k)
        print()
        _print_header()
        _benchmark_kernel(fn, args_lists, m, n, k, warmup=warmup, iters=iters)
    except ConfigError as e:
        print(f"  Skipped: {e}")

    if pytorch:
        print()

        # Benchmark PyTorch
        print("PyTorch matmul:")
        print(f"{'m':>6} {'n':>6} {'k':>6} {'ms':>8} {'TFLOP/s':>8}")
        args_lists = _make_torch_args_lists(m, n, k)
        _benchmark_kernel(
            torch.matmul,
            args_lists,
            m,
            n,
            k,
            warmup=warmup,
            iters=iters,
            args_not_tuple=True,
        )


def _parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Benchmark MMA checkerboard kernel configurations."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all configurations (default behavior if no args).",
    )
    parser.add_argument(
        "-m", type=int, default=4096, help="M dimension (default: 4096)."
    )
    parser.add_argument(
        "-n", type=int, default=4096, help="N dimension (default: 4096)."
    )
    parser.add_argument(
        "-k", type=int, default=4096, help="K dimension (default: 4096)."
    )
    parser.add_argument(
        "--warp-m",
        type=int,
        default=64,
        help="Warp M tile size (default: 64). Valid: 32, 64.",
    )
    parser.add_argument(
        "--warp-n",
        type=int,
        default=64,
        help="Warp N tile size (default: 64). Valid: 32, 64.",
    )
    parser.add_argument(
        "--warps-m",
        type=int,
        default=4,
        help="Number of warps along M dimension (default: 4).",
    )
    parser.add_argument(
        "--warps-n",
        type=int,
        default=2,
        help="Number of warps along N dimension (default: 2).",
    )
    parser.add_argument(
        "--chunk-k16",
        type=int,
        default=2,
        help="K chunk size in units of 16 (default: 2). Valid: 1, 2, 4.",
    )
    parser.add_argument(
        "--unroll-depth",
        type=int,
        default=None,
        help="Main loop unroll depth (default: None for full unroll).",
    )
    parser.add_argument(
        "--wave-size",
        type=int,
        default=4,
        help="Number of blocks in a wave along M dimension (default: 4).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100).",
    )
    parser.add_argument(
        "--pytorch",
        action="store_true",
        help="Also benchmark PyTorch matmul for comparison.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.all:
        _benchmark()
    else:
        config = MmaConfig(
            warp_m=args.warp_m,
            warp_n=args.warp_n,
            warps_m=args.warps_m,
            warps_n=args.warps_n,
            chunk_k16=args.chunk_k16,
            unroll_depth=args.unroll_depth,
            wave_size=args.wave_size,
        )
        _benchmark_single(
            args.m,
            args.n,
            args.k,
            config,
            warmup=args.warmup,
            iters=args.iters,
            pytorch=args.pytorch,
        )
