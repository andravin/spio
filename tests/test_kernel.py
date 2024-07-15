"""Unit tests that compile and test CUDA kernels that use tensor cores."""

import sys
from tempfile import TemporaryDirectory
from pathlib import Path

import cupy as cp
import torch
from torch import nn

from spio import (
    compile_test_kernel,
    generate_indices,
    IndexSpec,
    generate_tensors,
    TensorSpec,
    generate_params,
)

ADA_ARCH = "sm_89"


def _set_printoptions():
    """Set the CuPy printoptions to show full matrices."""
    cp.set_printoptions(linewidth=200, threshold=sys.maxsize)


def test_add_kernel():
    """Compile and run a simple CUDA kernel."""
    module, add_kernel = compile_test_kernel(kernel_name="add")

    x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    y = cp.zeros((5, 5), dtype=cp.float32)
    add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
    cp.testing.assert_array_equal(x1 + x2, y)


def test_mma_m16_n8_k8_kernel():
    """Compile and run a GPU kernel that tests tensor core mma with shape m16_n8_k8."""
    module, mma_kernel = compile_test_kernel(
        kernel_name="mma_m16_n8_k8", source_file_name="mma"
    )

    A = cp.zeros((16, 8), dtype=cp.float16)

    for i in range(16):
        for k in range(8):
            A[i, k] = (i * 8 + k) % 17

    B = cp.zeros((8, 8), dtype=cp.float16)
    for k in range(8):
        for j in range(8):
            B[k, j] = (k * 8 + j) % 19

    C = cp.zeros((16, 8), dtype=cp.float32)

    B_trans = cp.ascontiguousarray(cp.transpose(B))
    mma_kernel((1,), (32,), (C, A, B_trans))

    C_ref = cp.matmul(A.astype(cp.float32), B.astype(cp.float32))

    cp.testing.assert_array_equal(C_ref, C)


def test_mma_m16_n8_k16_kernel():
    """Compile and run a GPU kernel that tests tensor core mma with shape m16_n8_k16."""
    module, mma_kernel = compile_test_kernel(
        kernel_name="mma_m16_n8_k16", source_file_name="mma", debug=True
    )

    A = cp.zeros((16, 16), dtype=cp.float16)

    for i in range(16):
        for k in range(16):
            A[i, k] = (i * 16 + k) % 17

    B = cp.zeros((16, 8), dtype=cp.float16)
    for k in range(16):
        for j in range(8):
            B[k, j] = (k * 8 + j) % 19

    C = cp.zeros((16, 8), dtype=cp.float32)

    B_trans = cp.ascontiguousarray(cp.transpose(B))
    mma_kernel((1,), (32,), (C, A, B_trans))

    C_ref = cp.matmul(A.astype(cp.float32), B.astype(cp.float32))

    cp.testing.assert_array_equal(C_ref, C)


def test_conv_group_4_16w_4h_64c():
    N = 1
    C = 64
    H = 4
    W = 16
    R = 3
    S = 3
    K = C
    group_width = 8
    PADDING = 1

    groups = C // group_width

    BLOCK_W = W + 2
    P = H
    Q = W

    C8 = C // 8
    C2 = C // 2

    WARPS = C8
    THREADS = WARPS * 32

    conv = nn.Conv2d(C, K, 3, bias=False, padding=PADDING, groups=groups)
    weights = torch.randn((K, group_width, R, S))
    with torch.no_grad():
        conv.weight.copy_(weights)
    conv = conv.cuda()
    conv = conv.to(memory_format=torch.channels_last)

    inputs = torch.randn((N, C, H, W), device="cuda", dtype=torch.float16)
    inputs = inputs.to(memory_format=torch.channels_last)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            ref_outputs = conv(inputs)

    with TemporaryDirectory(prefix="spio_") as include_dir:
        index_header_file = Path(include_dir) / "my_indices.h"
        tensor_header_file = Path(include_dir) / "my_tensors.h"
        params_file = Path(include_dir) / "my_params.h"
        tiles_file = Path(include_dir) / "my_tiles.h"
        generate_indices(
            [
                IndexSpec("ThreadIdx", dict(warp=WARPS, lane=32)),
                IndexSpec("InputIdx", dict(n=N, h=H, w=W, c8=C8)),
                IndexSpec("OutputIdx", dict(n=N, p=H, q=W, c2=C2)),
                IndexSpec("SmemInputIdx", dict(y=H, x=BLOCK_W, c8=C8)),
                IndexSpec("SmemWeightsIdx", dict(k=K, r=R, s=S)),
                IndexSpec("WeightsOutIdx", dict(k=K, rs=R * S, c2=4)),
            ],
            index_header_file,
        )
        generate_tensors(
            [
                TensorSpec("Input", "const uint4", dict(n=N, h=H, w=W, c8=C8)),
                TensorSpec(
                    "ConstSmemInput", "const uint4", dict(y=H, x=BLOCK_W, c8=C8)
                ),
                TensorSpec("SmemInput", "uint4", dict(y=H, x=BLOCK_W, c8=C8)),
                TensorSpec("Output", "__half2", dict(n=N, p=H, q=W, c2=C2)),
                TensorSpec("ConstSmemWeights", "const uint4", dict(k=K, r=R, s=S)),
            ],
            tensor_header_file,
        )
        generate_params(
            "MyParams",
            dict(
                C=C,
                R=R,
                S=S,
                C8=C8,
                GROUP_WIDTH=group_width,
                P=P,
                Q=Q,
                PADDING=PADDING,
                WARPS=WARPS,
                THREADS=THREADS,
            ),
            params_file,
        )
        generate_params(
            "MyTiles",
            dict(
                BLOCK_W=Q + 2,
                BLOCK_C8=WARPS,
            ),
            tiles_file,
        )
        module, conv_kernel = compile_test_kernel(
            kernel_name="conv_group_4_16w_4h_64c",
            source_file_name="conv_group_4",
            debug=True,
            includes=[include_dir],
        )

    outputs = torch.zeros((N, C, H, W), device="cuda", dtype=torch.float16).to(
        memory_format=torch.channels_last
    )

    cp_outputs = cp.asarray(outputs)
    cp_inputs = cp.asarray(inputs)
    cp_weights = cp.asarray(conv.weight.detach().type(torch.float16))
    cp_loaded_input = cp.zeros((N, H, W + 2, C), dtype=cp.float16)
    cp_loaded_weights = cp.zeros((K, R, S, group_width), dtype=cp.float16)

    conv_kernel(
        (1,),
        (THREADS,),
        (cp_outputs, cp_loaded_input, cp_loaded_weights, cp_inputs, cp_weights),
    )

    for n in range(N):
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    assert cp_loaded_input[n, y, x + 1, c] == cp_inputs[n, c, y, x]
    cp.testing.assert_array_equal(cp_loaded_input[:, :, 0, :], 0)
    cp.testing.assert_array_equal(cp_loaded_input[:, :, -1, :], 0)

    for k in range(K):
        for c in range(group_width):
            for r in range(R):
                for s in range(S):
                    assert cp_loaded_weights[k, r, s, c] == cp_weights[k, c, r, s]

    diff = ref_outputs - outputs
    torch.testing.assert_close(ref_outputs, outputs)

    # _set_printoptions()
    # print(cp_loaded_input)
