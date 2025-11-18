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

    _, mma_kernel_16c = compile_and_load_kernel(
        kernel_name="mma_checkerboard_16c",
        source_file_name="mma_checkerboard_16c.cu",
        header_dict={"parameters.h": generate(specs)},
        src_module="spio.src_tests",
        lineinfo=True,
        max_registers=max_registers,
    )
    mma_kernel_16c.launch(grid, block, (C, A_16c, B_16c))

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
    n8 = n // 8

    block_x8 = block_x // 8
    block_x16 = block_x // 16

    double_chunk = 2 * config.chunk_k16
    assert (
        k16 % double_chunk == 0
    ), f"Kernel requires K={k} to be a multiple of twice the chunk size {double_chunk}."

    tensor_a = Tensor("A", dtype.uint4, Dims(k16=k16, i=m, k8=2), constant=True)
    tensor_b = Tensor("B", dtype.uint4, Dims(k16=k16, j=n, k8=2), constant=True)
    tensor_c = Tensor("C", dtype.uint4, Dims(i=m, j8=n8))

    smem_tensor_a = Tensor(
        "SmemA",
        dtype.uint4,
        Dims(
            ping=2,
            k16=config.chunk_k16,
            i16=block_x16,
            checkers=32,
        ),
    )
    smem_tensor_b = Tensor(
        "SmemB",
        dtype.uint4,
        Dims(
            ping=2,
            k16=config.chunk_k16,
            j16=block_x16,
            checkers=32,
        ),
    )
    a_tile = Tensor("A_Tile", "_A", Dims(k16=config.chunk_k16, i16=warp_m16))
    b_tile = Tensor("B_Tile", "_B", Dims(k16=config.chunk_k16, j16=warp_n16))
    c_tile = Tensor("C_Tile", "_C", Dims(i16=warp_m16, j16=warp_n16))

    specs = [
        Fold("block_i", "i", block_x),
        Fold("block_j", "j", block_x),
        Fold("warp_i", "i", config.warp_m),
        Fold("warp_j", "j", config.warp_n),
        tensor_a,
        tensor_b,
        tensor_c,
        Index("GlobalLoadIndex", Dims(x16=block_x16, x=16, k8=2)),
        Index("ComputeIndex", Dims(warp_i=warps_m, warp_j=warps_n, lane=32)),
        smem_tensor_a,
        smem_tensor_b,
        AsyncStripLoader(
            "A_Loader",
            smem_tensor=smem_tensor_a,
            gmem_tensor=tensor_a,
            minor_axis="k16",
            major_axis_size=block_x16,
            minor_axis_size=config.chunk_k16,
            num_warps=warps,
        ),
        AsyncStripLoader(
            "B_Loader",
            smem_tensor=smem_tensor_b,
            gmem_tensor=tensor_b,
            minor_axis="k16",
            major_axis_size=block_x16,
            minor_axis_size=config.chunk_k16,
            num_warps=warps,
        ),
        Fragment("_A", FragmentType.M16_K16_F16_A, "i", "k"),
        Fragment("_B", FragmentType.N16_K16_F16_B, "k", "j"),
        Fragment("_C", FragmentType.M16_N16_F32_C, "i", "j"),
        a_tile,
        b_tile,
        c_tile,
        Matmul(a_tile, b_tile, c_tile, c_tile, function_name="mma"),
        Index("SmemCLoadIndex", Dims(i=32, j8=8)),
        Checkerboard("Smem_Checkers", "x", "k8", "checkers"),
        Checkerboard("SmemA_Checkers", "i", "k8", "checkers"),
        Checkerboard("SmemB_Checkers", "j", "k8", "checkers"),
        Tensor(
            "SmemCStore",
            dtype.half2,
            Dims(warp_i=warps_m, j8=block_x8, i16=warp_m16, i=16, j2=4),
            strides=Strides(j8=(32 + 1) * 4),
        ),
        Tensor(
            "SmemCLoad",
            dtype.uint4,
            Dims(warp_i=warps_m, j8=block_x8, i=32),
            Strides(j8=32 + 1),
            constant=True,
        ),
    ]

    if warps == 8:
        max_registers = 128
    elif warps == 4:
        max_registers = 255
    else:
        raise ValueError(f"Unsupported number of warps: {warps}")

    return (specs, grid, block, max_registers)
