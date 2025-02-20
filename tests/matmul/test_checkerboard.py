"""Test matrix multiplication using the checkerboard shared-memory layout."""

import torch

from spio.generators import (
    IndexSpec,
    TensorSpec,
    ParamsSpec,
    FragmentSpec,
    CheckerboardIndexSpec,
    AsyncStripLoaderSpec,
    FoldSpec,
    generate,
)
from spio.compiler import compile_and_load_kernel
from spio.util import divup, assert_all_close_with_acc_depth, SixteenChannelsLast
from spio.util.test_matrices import make_test_matrices, matmul_trans_ref


def _make_checkerboard_kernel_params(
    m: int, n: int, k: int, sixteen_channels_last: bool = False
):

    assert k % 16 == 0, "Kernel requires K to be a multiple of 16."

    block_x = 128

    warp_m = 32
    warp_n = 64

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
    k8 = k // 8
    n8 = n // 8

    block_x16 = block_x // 16
    block_x32 = block_x // 32
    block_x64 = block_x // 64

    if sixteen_channels_last:
        tensor_a = TensorSpec("A", "const uint4", dict(k16=k16, i=m, k8=2))
        tensor_b = TensorSpec("B", "const uint4", dict(k16=k16, j=n, k8=2))
        chunk_k16 = 2
    else:
        tensor_a = TensorSpec("A", "const uint4", dict(i=m, k8=k8))
        tensor_b = TensorSpec("B", "const uint4", dict(j=n, k8=k8))
        chunk_k16 = 1

    assert k16 % chunk_k16 == 0, "Kernel requires K to be a multiple of the chunk size."

    smem_tensor_a = TensorSpec(
        "SmemA",
        "uint4",
        dict(
            ping_pong=2,
            k16=chunk_k16,
            i32=block_x32,
            i16=2,
            checkers=CheckerboardIndexSpec(i=16, k8=2),
        ),
    )
    smem_tensor_b = TensorSpec(
        "SmemB",
        "uint4",
        dict(
            ping_pong=2,
            k16=chunk_k16,
            j64=block_x64,
            j16=4,
            checkers=CheckerboardIndexSpec(j=16, k8=2),
        ),
    )

    specs = [
        FoldSpec("block_i", "i", block_x),
        FoldSpec("block_j", "j", block_x),
        ParamsSpec("Params", params),
        tensor_a,
        tensor_b,
        TensorSpec("C", "uint4", dict(i=m, j8=n8)),
        IndexSpec("GlobalLoadIndex", dict(x=block_x, k8=2)),
        IndexSpec("ComputeIndex", dict(i32=4, j64=2, lane=32)),
        smem_tensor_a,
        smem_tensor_b,
        TensorSpec(
            "SmemCStore",
            "__half2",
            dict(i32=4, j64=2, j16=4, j8=2, i16=2, i=16, j2=4),
            strides=dict(j8=(32 + 1) * 4),
        ),
        TensorSpec(
            "SmemCLoad",
            "const uint4",
            dict(i32=4, j64=2, j8=8, i=32),
            strides=dict(j8=32 + 1),
        ),
        IndexSpec("SmemCLoadIndex", dict(i=32, j8=8)),
        FragmentSpec("A_Fragment", "MMA_M16_K16_F16_A", "i", "k"),
        FragmentSpec("B_Fragment", "MMA_N16_K16_F16_B", "k", "j"),
        FragmentSpec("C_Fragment", "MMA_M16_N16_F32_C", "i", "j"),
        TensorSpec("C_Tensor", "C_Fragment", dict(i16=warp_m16, j16=warp_n16)),
        TensorSpec("A_Tensor", "A_Fragment", dict(k16=chunk_k16, i16=warp_m16)),
        TensorSpec("B_Tensor", "B_Fragment", dict(k16=chunk_k16, j16=warp_n16)),
    ]

    if sixteen_channels_last:
        specs += [
            AsyncStripLoaderSpec(
                "A_Loader",
                smem_tensor=smem_tensor_a,
                gmem_tensor=tensor_a,
                minor_axis="k16",
                major_axis_size=block_x16,
                minor_axis_size=chunk_k16,
                num_warps=warps,
            ),
            AsyncStripLoaderSpec(
                "B_Loader",
                smem_tensor=smem_tensor_b,
                gmem_tensor=tensor_b,
                minor_axis="k16",
                major_axis_size=block_x16,
                minor_axis_size=chunk_k16,
                num_warps=warps,
            ),
        ]

    return (specs, grid, block)


def test_mma_checkerboard_kernel():
    """Compile and run a GPU kernel that tests tensor core mma with checkerboard layout for smem."""

    m = 512
    n = 256
    k = 128

    specs, grid, block = _make_checkerboard_kernel_params(
        m, n, k, sixteen_channels_last=False
    )

    params_header = generate(specs)

    A, B_trans, C = make_test_matrices(
        m=m, n=n, k=k, ones=False, output_dtype=torch.float16
    )
    C_ref = matmul_trans_ref(A, B_trans)

    _, mma_kernel = compile_and_load_kernel(
        kernel_name="mma_checkerboard",
        source_file_name="mma_checkerboard.cu",
        debug=False,
        lineinfo=True,
        header_dict={"parameters.h": params_header},
        src_module="spio.src_tests",
    )
    mma_kernel.launch(grid, block, (C, A, B_trans))

    assert_all_close_with_acc_depth(C, C_ref, acc_depth=k)


def test_mma_checkerboard_16c_kernel():
    """Compile and run a GPU kernel that tests tensor core mma with checkerboard layout for smem."""

    big_test = False

    if big_test:
        m, n, k = (4096, 4096, 1024)
    else:
        m, n, k = (512, 256, 128)

    specs, grid, block = _make_checkerboard_kernel_params(
        m, n, k, sixteen_channels_last=True
    )

    params_header = generate(specs)

    A, B_trans, C = make_test_matrices(
        m=m, n=n, k=k, ones=False, output_dtype=torch.float16
    )

    A_16c = SixteenChannelsLast.to(A)
    B_16c = SixteenChannelsLast.to(B_trans)

    _, mma_kernel_16c = compile_and_load_kernel(
        kernel_name="mma_checkerboard_16c",
        source_file_name="mma_checkerboard_16c.cu",
        debug=False,
        lineinfo=True,
        header_dict={"parameters.h": params_header},
        src_module="spio.src_tests",
    )
    mma_kernel_16c.launch(grid, block, (C, A_16c, B_16c))

    C_ref = matmul_trans_ref(A, B_trans)
    assert_all_close_with_acc_depth(C, C_ref, acc_depth=k)
