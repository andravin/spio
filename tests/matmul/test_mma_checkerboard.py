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

    m_warp: int = 32
    n_warp: int = 64
    m_warps: int = 4
    n_warps: int = 2
    k16_chunk: int = 2
    wave_size: int = 4

    unroll_depth: int = 1

    @property
    def unroll_depth_str(self) -> str:
        """Return a string representation of the unroll depth."""
        if self.unroll_depth is None:
            return ""
        return str(self.unroll_depth)

    @property
    def m_block(self) -> int:
        """Return the block M size."""
        return self.m_warp * self.m_warps

    @property
    def n_block(self) -> int:
        """Return the block N size."""
        return self.n_warp * self.n_warps

    @property
    def warps(self) -> int:
        """Return the total number of warps."""
        return self.m_warps * self.n_warps

    @property
    def k_chunk(self) -> int:
        """Return the K chunk size."""
        return self.k16_chunk * 16


WAVE_SIZE = [2, 4, 8]

BASE_CONFIGS = [
    MmaConfig(m_warp=64, n_warp=32, m_warps=2, n_warps=4, k16_chunk=2),
    MmaConfig(m_warp=64, n_warp=32, m_warps=2, n_warps=4, k16_chunk=4),
    MmaConfig(m_warp=64, n_warp=64, m_warps=2, n_warps=4, k16_chunk=2),
    MmaConfig(m_warp=64, n_warp=64, m_warps=2, n_warps=4, k16_chunk=4),
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

    # Validate problem size.
    assert k % 128 == 0, "Kernel requires K to be a multiple of 128."
    assert n % 128 == 0, "Kernel requires N to be a multiple of 128."
    assert m % 128 == 0, "Kernel requires M to be a multiple of 128."
    if k % config.k_chunk != 0:
        raise ConfigError(
            f"K ({k}) must be a multiple of k_chunk ({config.k_chunk}) for this config."
        )

    # Base dimensions
    I = Dim()
    J = Dim()
    K = Dim()

    # Matrix dimensions.
    i = I(m)
    j = J(n)
    k = K(k)

    # Define the thread-block traversal order.
    BlockIdx = CompoundIndex(
        Dims(
            i / (config.wave_size * config.m_block),
            j / config.n_block,
            (i / config.m_block) % config.wave_size,
        ),
        init=BuiltIn.BLOCK_IDX_X,
    )

    # Block dimensions.
    i_block = I(config.m_block)
    j_block = J(config.n_block)

    # Warp dimensions.
    i_warp = I(config.m_warp)
    j_warp = J(config.n_warp)

    # Number of warps.
    i_warps = I(config.m_block) / config.m_warp
    j_warps = J(config.n_block) / config.n_warp

    # Add to the Generators container any classes that the kernel will use by name.
    g = Generators()

    # K8 becomes a CUDA/C++ class that the kernel can use.
    K8 = g.K8 = K / 8

    # Load indices
    ALoadGlobalIndex = CompoundIndex(Dims(i_block, K8(2)), init=BuiltIn.THREAD_IDX_X)
    BLoadGlobalIndex = CompoundIndex(Dims(j_block, K8(2)), init=BuiltIn.THREAD_IDX_X)

    # The position of each warp's tile and the LANE dimension for each thread.
    LocalIndex = CompoundIndex(
        Dims(i_warps, j_warps, LANE(32)), init=BuiltIn.THREAD_IDX_X
    )

    # Global memory tensors
    g.AGlobal = Tensor(
        dtype.half8, Dims(i / 16, k / 16, i % 16, (k % 16) / 8), constant=True
    )[ALoadGlobalIndex, BlockIdx]
    g.BGlobal = Tensor(
        dtype.half8, Dims(j / 16, k / 16, j % 16, (k % 16) / 8), constant=True
    )[BLoadGlobalIndex, BlockIdx]
    g.CGlobal = Tensor(dtype.half8, Dims(i / 16, j / 16, i % 16, (j % 16) / 8))[
        BlockIdx, LocalIndex
    ]

    # Depth of a single inner loop iteration.
    g.K16 = K / 16
    k16_chunk = g.K16(config.k16_chunk)

    # Shared memory tensors
    g.ASmem = Tensor(
        dtype.half8,
        Dims(
            k16_chunk * 2,
            i_block / 16,
            swizzle=Checkerboard(pairs_dim=I, colors_dim=K / 8, size=32),
        ),
    )
    g.BSmem = Tensor(
        dtype.half8,
        Dims(
            k16_chunk * 2,
            j_block / 16,
            swizzle=Checkerboard(pairs_dim=J, colors_dim=K / 8, size=32),
        ),
    )
    J8 = J / 8
    g.CSmem = Tensor(
        dtype.half2,
        Dims(i_warps, j_block / 8, i_warp, (j_block % 8) / 2),
        strides=Strides(J8((config.m_warp + 1) * 4)),
    )

    smem_size = max(g.ASmem.num_bytes + g.BSmem.num_bytes, g.CSmem.num_bytes)
    smem_capacity = driver.get_max_shared_memory_per_block_optin()

    if smem_size > smem_capacity:
        raise ConfigError(
            f"Required shared memory size {smem_size} exceeds device capacity {smem_capacity}"
        )

    # MMA fragments
    g.AFragment = Fragment(FragmentType.M16_K16_F16_A, I, K)
    g.BFragment = Fragment(FragmentType.N16_K16_F16_B, K, J)
    g.CFragment = Fragment(FragmentType.M16_N16_F32_C, I, J)

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
        inner_axis=i_block / 16,
        outer_axis=k16_chunk,
        num_warps=config.warps,
        num_buffers=2,
    )
    g.BLoader = AsyncStripLoader(
        smem_tensor=g.BStoreSmem,
        gmem_tensor=g.BGlobal,
        inner_axis=j_block / 16,
        outer_axis=k16_chunk,
        num_warps=config.warps,
        num_buffers=2,
    )

    # Local memory tensors (i.e. registers)
    # - Each tensor "element" is itself a matrix fragment.
    g.AReg = Tensor(g.AFragment, Dims(k16_chunk, i_warp / 16))
    g.BReg = Tensor(g.BFragment, Dims(k16_chunk, j_warp / 16))
    g.CReg = Tensor(g.CFragment, Dims(i_warp / 16, j_warp / 16))

    # Matmul operation
    g.mma = Matmul(g.AReg, g.BReg, g.CReg, g.CReg)

    # Transpose output through shared memory
    g.CStoreSmem = g.CSmem.with_dim(g.CFragment.compound_index)[LocalIndex]
    g.CLoadSmem = g.CSmem.vector_length(8, constant=True)[LocalIndex]
    g.CLoadSmemIndex = CompoundIndex(Dims(i_warp, j_warp / 8))

    g.c_output_idx = g.CLoadSmemIndex.partition(LANE, LocalIndex)

    # Macro for loop unrolling
    g.macros = Macro(dict(MAIN_LOOP_UNROLL_DEPTH=config.unroll_depth_str))

    # Register dimensions ..
    g.K = K

    # .. and folded dimensions that are used by name in the kernel.
    g.K_CHUNK = K / config.k_chunk
    g.K_DOUBLE_CHUNK = g.K_CHUNK / 2

    # Calculate the block size.
    threads = config.warps * 32
    block = (threads, 1, 1)

    # Calculate the grid size.
    i_blocks = divup(m, config.m_block)
    j_blocks = divup(n, config.n_block)
    blocks = i_blocks * j_blocks
    grid = (blocks, 1, 1)

    # Determine max registers per thread.
    num_warp_elements = config.m_warp * config.n_warp
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
        f"Config: warp_m={config.m_warp}, warp_n={config.n_warp}, "
        f"chunk_k16={config.k16_chunk}, unroll_depth={config.unroll_depth}"
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
            m_warp=args.warp_m,
            n_warp=args.warp_n,
            m_warps=args.warps_m,
            n_warps=args.warps_n,
            k16_chunk=args.chunk_k16,
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
