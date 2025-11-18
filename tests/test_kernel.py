"""Unit tests that compile and test CUDA kernels that use tensor cores."""

import torch

from spio.generators import *
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


def test_row_memcpy_kernel():
    """Test the row-by-row memcpy kernel."""
    debug = False
    lineinfo = True

    # Parameters
    # The kernel achieves 92% of peak DRAM memory bandwidth with N=128 C=32 H=64 W=64.
    N = 128
    C = 32
    H = 64
    W = 64

    # Hardcoded parameter:
    GROUP_WIDTH = 4

    # Derived parameters
    C4 = C // 4
    GROUPS = C // GROUP_WIDTH

    # Tiles
    BLOCK_P = min(16, H)
    BLOCK_Q = 16
    BLOCK_GROUPS = min(8, GROUPS)

    # Derived Tiles
    BLOCK_C = BLOCK_GROUPS * GROUP_WIDTH
    BLOCK_C4 = BLOCK_C // 4
    BLOCK_W = BLOCK_Q + 2
    BLOCKS_N = N
    BLOCKS_P = divup(H, BLOCK_P)
    BLOCKS_Q = divup(W, BLOCK_Q)
    BLOCKS_C4 = divup(C4, BLOCK_C4)
    BLOCKS = BLOCKS_N * BLOCKS_P * BLOCKS_Q * BLOCKS_C4
    WARPS = BLOCK_GROUPS
    THREADS = WARPS * 32

    # Generate code specifications
    specs = [
        # Fold dimensions
        Fold("block_p", "p", BLOCK_P),
        Fold("block_q", "q", BLOCK_Q),
        Fold("block_c", "c", BLOCK_C),
        # Parameters
        ParamsSpec("Block", {"padding": 1, "c4": BLOCK_C4}),
        # Index types
        Index(
            "BlockIdx",
            Dims(n=BLOCKS_N, block_p=BLOCKS_P, block_q=BLOCKS_Q, block_c=BLOCKS_C4),
        ),
        Index("InputIdx", Dims(x=BLOCK_W, c4=BLOCK_C4)),
        # Tensor types
        Tensor(
            "Input",
            dtype.float4,
            Dims(n=N, y=H, x=W, c4=C4),
            constant=True,
        ),
        Tensor("Output", dtype.float4, Dims(n=N, p=H, q=W, c4=C4)),
        Tensor(
            "SmemInput",
            dtype.float4,
            Dims(ping_pong=2, x=BLOCK_W, c4=BLOCK_C4 + 1),
        ),
        Tensor(
            "ConstSmemInput",
            dtype.float2,
            Dims(ping_pong=2, x=BLOCK_W, c4=BLOCK_C4 + 1, c2=2),
            constant=True,
        ),
        Index("SmemInputLoadIdx", Dims(c4=BLOCK_C4, q=BLOCK_Q, c2=2)),
        Tensor(
            "SmemOutput",
            dtype.float2,
            Dims(q=BLOCK_Q, c4=BLOCK_C4 + 1, c2=2),
        ),
        Tensor(
            "ConstSmemOutput",
            dtype.float4,
            Dims(q=BLOCK_Q, c4=BLOCK_C4 + 1),
            constant=True,
        ),
    ]
    parameters_header = generate(specs)

    # Compile the kernel with our generated headers
    _, kernel = compile_and_load_kernel(
        kernel_name="row_memcpy",
        debug=debug,
        lineinfo=lineinfo,
        header_dict={"parameters.h": parameters_header},
        src_module="spio.src_tests",
    )

    # Create test tensors
    inputs = torch.randn((N, C, H, W), device="cuda", dtype=torch.float32).to(
        memory_format=torch.channels_last
    )
    outputs = torch.zeros((N, C, H, W), device="cuda", dtype=torch.float32).to(
        memory_format=torch.channels_last
    )

    # Run kernel and verify results
    kernel.launch((BLOCKS, 1, 1), (THREADS, 1, 1), (outputs, inputs))
    assert torch.equal(outputs, inputs)
