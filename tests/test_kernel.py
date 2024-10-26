"""Unit tests that compile and test CUDA kernels that use tensor cores."""

import torch

from spio.generators import IndexSpec, TensorSpec, ParamsSpec, generate
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


def test_mma_m16_n8_k8_kernel():
    """Compile and run a GPU kernel that tests tensor core mma with shape m16_n8_k8."""
    _, mma_kernel = compile_and_load_kernel(
        kernel_name="mma_m16_n8_k8",
        source_file_name="mma.cu",
        src_module="spio.src_tests",
    )

    A = torch.zeros((16, 8), dtype=torch.float16, device="cuda")

    for i in range(16):
        for k in range(8):
            A[i, k] = (i * 8 + k) % 17

    B = torch.zeros((8, 8), dtype=torch.float16, device="cuda")
    for k in range(8):
        for j in range(8):
            B[k, j] = (k * 8 + j) % 19

    C = torch.zeros((16, 8), dtype=torch.float32, device="cuda")

    B_trans = torch.transpose(B, 0, 1).contiguous()
    mma_kernel.launch((1, 1, 1), (32, 1, 1), (C, A, B_trans))

    C_ref = torch.matmul(A.float(), B.float())

    assert_all_close_with_acc_depth(C, C_ref, acc_depth=8)


def test_mma_m16_n8_k16_kernel():
    """Compile and run a GPU kernel that tests tensor core mma with shape m16_n8_k16."""
    _, mma_kernel = compile_and_load_kernel(
        kernel_name="mma_m16_n8_k16",
        source_file_name="mma.cu",
        debug=True,
        src_module="spio.src_tests",
    )

    A = torch.zeros((16, 16), dtype=torch.float16, device="cuda")

    for i in range(16):
        for k in range(16):
            A[i, k] = (i * 16 + k) % 17

    B = torch.zeros((16, 8), dtype=torch.float16, device="cuda")
    for k in range(16):
        for j in range(8):
            B[k, j] = (k * 8 + j) % 19

    C = torch.zeros((16, 8), dtype=torch.float32, device="cuda")

    B_trans = torch.transpose(B, 0, 1).contiguous()
    mma_kernel.launch((1, 1, 1), (32, 1, 1), (C, A, B_trans))

    C_ref = torch.matmul(A.float(), B.float())

    assert_all_close_with_acc_depth(C, C_ref, acc_depth=16)


def test_memcpy_kernel():
    """This kernel achives 92% of peak DRAM memory bandwidth on NVIDIA RTX 4090."""
    debug = False
    lineinfo = True

    N = 128
    C = 32
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

    parameters_header = generate(
        [
            ParamsSpec(
                "Block", {"p": BLOCK_P, "q": BLOCK_Q, "c4": BLOCK_C4, "padding": 1}
            ),
            IndexSpec(
                "BlockIdx",
                {"n": BLOCKS_N, "p": BLOCKS_P, "q": BLOCKS_Q, "c4": BLOCKS_C4},
            ),
            IndexSpec("InputIdx", {"x": BLOCK_W, "c4": BLOCK_C4}),
            TensorSpec("Input", "const float4", {"n": N, "y": H, "x": W, "c4": C4}),
            TensorSpec("Output", "float4", {"n": N, "p": H, "q": W, "c4": C4}),
            TensorSpec(
                "SmemInput",
                "float4",
                {"ping_pong": 2, "x": BLOCK_W, "c4": BLOCK_C4 + 1},
            ),
            TensorSpec(
                "ConstSmemInput",
                "const float2",
                {"ping_pong": 2, "x": BLOCK_W, "c4": BLOCK_C4 + 1, "c2": 2},
            ),
            IndexSpec("SmemInputLoadIdx", {"c4": BLOCK_C4, "q": BLOCK_Q, "c2": 2}),
            TensorSpec(
                "SmemOutput",
                "float2",
                {"q": BLOCK_Q, "c4": BLOCK_C4 + 1, "c2": 2},
            ),
            TensorSpec(
                "ConstSmemOutput",
                "const float4",
                {"q": BLOCK_Q, "c4": BLOCK_C4 + 1},
            ),
        ]
    )

    _, kernel = compile_and_load_kernel(
        kernel_name="row_memcpy",
        debug=debug,
        lineinfo=lineinfo,
        header_dict={"parameters.h": parameters_header},
        src_module="spio.src_tests",
    )

    inputs = torch.randn((N, C, H, W), device="cuda", dtype=torch.float32).to(
        memory_format=torch.channels_last
    )

    outputs = torch.zeros((N, C, H, W), device="cuda", dtype=torch.float32).to(
        memory_format=torch.channels_last
    )

    kernel.launch((BLOCKS, 1, 1), (THREADS, 1, 1), (outputs, inputs))
    assert torch.equal(outputs, inputs)
