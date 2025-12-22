"""Unit tests that compile and test CUDA kernels that use tensor cores."""

import torch

from spio.generators import generate, ParamsSpec
from spio.compiler import compile_and_load_kernel
from spio.util import divup, assert_all_close_with_acc_depth


def test_add_kernel():
    """Compile and run a simple CUDA kernel."""
    _, add_kernel = compile_and_load_kernel(
        kernel_name="add", src_module="spio.src_tests"
    )

    x1 = torch.arange(25, dtype=torch.float32, device="cuda").reshape(5, 5)
    x2 = torch.arange(25, dtype=torch.float32, device="cuda").reshape(5, 5)
    y = torch.zeros((5, 5), dtype=torch.float32, device="cuda")
    add_kernel.launch((5, 1, 1), (5, 1, 1), (x1, x2, y))  # grid, block and arguments
    assert_all_close_with_acc_depth(y, x1 + x2, acc_depth=25)


def test_memcpy_kernel():
    """This kernel achives 92% of peak DRAM memory bandwidth on NVIDIA RTX 4090."""
    debug = False
    lineinfo = True

    N = 128
    C = 128
    H = 64
    W = 64

    WARPS = 8
    THREADS = WARPS * 32

    ITERS = 16
    VECTOR_DIM = 4

    BLOCK_X = ITERS * THREADS * VECTOR_DIM
    BLOCK_X4 = BLOCK_X // 4

    X = N * C * H * W
    BLOCKS = divup(X, BLOCK_X)

    my_params_header = generate(
        [
            ParamsSpec(
                "MyParams",
                {"ITERS": ITERS, "BLOCK_X4": BLOCK_X4, "X": X, "THREADS": THREADS},
            ),
        ]
    )

    _, memcpy_kernel = compile_and_load_kernel(
        kernel_name="memcpy_simple",
        debug=debug,
        lineinfo=lineinfo,
        header_dict={"my_params.h": my_params_header},
        src_module="spio.src_tests",
    )

    inputs = torch.randn((N, C, H, W), device="cuda", dtype=torch.float32).to(
        memory_format=torch.channels_last
    )

    outputs = torch.zeros((N, C, H, W), device="cuda", dtype=torch.float32).to(
        memory_format=torch.channels_last
    )

    memcpy_kernel.launch((BLOCKS, 1, 1), (THREADS, 1, 1), (outputs, inputs))

    assert torch.equal(outputs, inputs)


def test_index():
    """Test the index class."""
    debug = False
    lineinfo = True

    _, index = compile_and_load_kernel(
        kernel_name="index",
        debug=debug,
        lineinfo=lineinfo,
        src_module="spio.src_tests",
    )

    BLOCKS = 1
    THREADS = 256
    I = 64
    J = 4

    inputs = torch.randn((I, J), device="cuda", dtype=torch.float32)
    outputs = torch.zeros((I, J), device="cuda", dtype=torch.float32)

    index.launch((BLOCKS, 1, 1), (THREADS, 1, 1), (outputs, inputs))
    assert torch.equal(outputs, inputs)
