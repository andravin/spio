"""Test matrix multiplication using the checkerboard shared-memory layout."""

from dataclasses import dataclass


import torch

import spio.generators as gen
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
        header_dict={"parameters.h": gen.generate(specs)},
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

    tensor_a = gen.Tensor(
        "A", gen.dtype.uint4, gen.Dims(k16=k16, i=m, k8=2), constant=True
    )
    tensor_b = gen.Tensor(
        "B", gen.dtype.uint4, gen.Dims(k16=k16, j=n, k8=2), constant=True
    )
    tensor_c = gen.Tensor("C", gen.dtype.uint4, gen.Dims(i=m, j8=n8))

    smem_tensor_a = gen.Tensor(
        "SmemA",
        gen.dtype.uint4,
        gen.Dims(
            ping=2,
            k16=config.chunk_k16,
            i16=block_x16,
            checkers=32,
        ),
    )
    smem_tensor_b = gen.Tensor(
        "SmemB",
        gen.dtype.uint4,
        gen.Dims(
            ping=2,
            k16=config.chunk_k16,
            j16=block_x16,
            checkers=32,
        ),
    )
    a_tile = gen.Tensor("A_Tile", "_A", gen.Dims(k16=config.chunk_k16, i16=warp_m16))
    b_tile = gen.Tensor("B_Tile", "_B", gen.Dims(k16=config.chunk_k16, j16=warp_n16))
    c_tile = gen.Tensor("C_Tile", "_C", gen.Dims(i16=warp_m16, j16=warp_n16))

    specs = [
        gen.Fold("block_i", "i", block_x),
        gen.Fold("block_j", "j", block_x),
        gen.Fold("warp_i", "i", config.warp_m),
        gen.Fold("warp_j", "j", config.warp_n),
        tensor_a,
        tensor_b,
        tensor_c,
        gen.Index("GlobalLoadIndex", gen.Dims(x16=block_x16, x=16, k8=2)),
        gen.Index("ComputeIndex", gen.Dims(warp_i=warps_m, warp_j=warps_n, lane=32)),
        smem_tensor_a,
        smem_tensor_b,
        gen.AsyncStripLoader(
            "A_Loader",
            smem_tensor=smem_tensor_a,
            gmem_tensor=tensor_a,
            minor_axis="k16",
            major_axis_size=block_x16,
            minor_axis_size=config.chunk_k16,
            num_warps=warps,
        ),
        gen.AsyncStripLoader(
            "B_Loader",
            smem_tensor=smem_tensor_b,
            gmem_tensor=tensor_b,
            minor_axis="k16",
            major_axis_size=block_x16,
            minor_axis_size=config.chunk_k16,
            num_warps=warps,
        ),
        gen.Fragment("_A", gen.FragmentType.M16_K16_F16_A, "i", "k"),
        gen.Fragment("_B", gen.FragmentType.N16_K16_F16_B, "k", "j"),
        gen.Fragment("_C", gen.FragmentType.M16_N16_F32_C, "i", "j"),
        a_tile,
        b_tile,
        c_tile,
        gen.Matmul(a_tile, b_tile, c_tile, c_tile, function_name="mma"),
        gen.Index("SmemCLoadIndex", gen.Dims(i=32, j8=8)),
        gen.Checkerboard("Smem_Checkers", "x", "k8", "checkers"),
        gen.Checkerboard("SmemA_Checkers", "i", "k8", "checkers"),
        gen.Checkerboard("SmemB_Checkers", "j", "k8", "checkers"),
        gen.Tensor(
            "SmemCStore",
            gen.dtype.half2,
            gen.Dims(warp_i=warps_m, j8=block_x8, i16=warp_m16, i=16, j2=4),
            strides=gen.Strides(j8=(32 + 1) * 4),
        ),
        gen.Tensor(
            "SmemCLoad",
            gen.dtype.uint4,
            gen.Dims(warp_i=warps_m, j8=block_x8, i=32),
            gen.Strides(j8=32 + 1),
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
