"""Test matrix multiplication using the checkerboard shared-memory layout."""

import torch

import spio.generators as gen
from spio.compiler import compile_and_load_kernel
from spio.util import divup, assert_all_close_with_acc_depth, SixteenChannelsLast


def test_mma_checkerboard_16c_kernel():
    """Compile and run a GPU kernel that tests tensor core mma with checkerboard layout for smem."""

    m, n, k = (347, 248, 128)

    specs, grid, block = _get_specs(m, n, k)

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
    )
    mma_kernel_16c.launch(grid, block, (C, A_16c, B_16c))

    C_ref = torch.matmul(A, B_trans.transpose(0, 1))
    assert_all_close_with_acc_depth(C, C_ref, acc_depth=k)


def _get_specs(m: int, n: int, k: int):
    """Return the generator specs, grid and block for the mma checkerboard kernel."""
    assert k % 16 == 0, "Kernel requires K to be a multiple of 16."
    assert n % 8 == 0, "Kernel requires N to be a multiple of 8."

    block_x = 128
    warp_m = 32
    warp_n = 64
    chunk_k16 = 2

    warps_m = block_x // warp_m
    warps_n = block_x // warp_n
    warps = warps_m * warps_n
    threads = warps * 32

    blocks_m = divup(m, block_x)
    blocks_n = divup(n, block_x)

    grid = (blocks_m, blocks_n, 1)

    block = (threads, 1, 1)

    warp_m16 = warp_m // 16
    warp_n16 = warp_n // 16

    params = dict(
        warp_m16=warp_m16,
        warp_n16=warp_n16,
        k16=k // 16,
        vec_bytes=8 * 2,
        warps=warps,
    )
    k16 = k // 16
    n8 = n // 8

    block_x16 = block_x // 16
    block_x32 = block_x // 32
    block_x64 = block_x // 64

    assert k16 % chunk_k16 == 0, "Kernel requires K to be a multiple of the chunk size."

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
            ping_pong=2,
            k16=chunk_k16,
            i32=block_x32,
            i16=2,
            checkers=gen.CheckerboardIndex(i=16, k8=2),
        ),
    )
    smem_tensor_b = gen.Tensor(
        "SmemB",
        gen.dtype.uint4,
        gen.Dims(
            ping_pong=2,
            k16=chunk_k16,
            j64=block_x64,
            j16=4,
            checkers=gen.CheckerboardIndex(j=16, k8=2),
        ),
    )
    smem_tensor_c_store = gen.Tensor(
        "SmemCStore",
        gen.dtype.half2,
        gen.Dims(i32=4, j64=2, j16=4, j8=2, i16=2, i=16, j2=4),
        strides=gen.Strides(j8=(32 + 1) * 4),
    )
    smem_tensor_c_load = gen.Tensor(
        "SmemCLoad",
        gen.dtype.uint4,
        gen.Dims(i32=4, j64=2, j8=8, i=32),
        gen.Strides(j8=32 + 1),
        constant=True,
    )

    specs = [
        gen.Fold("block_i", "i", block_x),
        gen.Fold("block_j", "j", block_x),
        gen.ParamsSpec("Params", params),
        tensor_a,
        tensor_b,
        tensor_c,
        gen.Index("GlobalLoadIndex", gen.Dims(x=block_x, k8=2)),
        gen.Index("ComputeIndex", gen.Dims(i32=4, j64=2, lane=32)),
        smem_tensor_a,
        smem_tensor_b,
        gen.AsyncStripLoader(
            "A_Loader",
            smem_tensor=smem_tensor_a,
            gmem_tensor=tensor_a,
            minor_axis="k16",
            major_axis_size=block_x16,
            minor_axis_size=chunk_k16,
            num_warps=warps,
        ),
        gen.AsyncStripLoader(
            "B_Loader",
            smem_tensor=smem_tensor_b,
            gmem_tensor=tensor_b,
            minor_axis="k16",
            major_axis_size=block_x16,
            minor_axis_size=chunk_k16,
            num_warps=warps,
        ),
        gen.Fragment("_A", gen.FragmentType.M16_K16_F16_A, "i", "k"),
        gen.Fragment("_B", gen.FragmentType.N16_K16_F16_B, "k", "j"),
        gen.Fragment("_C", gen.FragmentType.M16_N16_F32_C, "i", "j"),
        gen.Tensor("A_Fragments", "_A", gen.Dims(k16=chunk_k16, i16=warp_m16)),
        gen.Tensor("B_Fragments", "_B", gen.Dims(k16=chunk_k16, j16=warp_n16)),
        gen.Tensor("C_Fragments", "_C", gen.Dims(i16=warp_m16, j16=warp_n16)),
        gen.Index("SmemCLoadIndex", gen.Dims(i=32, j8=8)),
        smem_tensor_c_store,
        smem_tensor_c_load,
    ]

    return (specs, grid, block)
