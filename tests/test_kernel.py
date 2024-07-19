"""Unit tests that compile and test CUDA kernels that use tensor cores."""

import cupy as cp
import torch
from torch import nn

from spio import (
    divup,
    GenDirectory,
    IndexSpec,
    TensorSpec,
    ParamsSpec,
    FragmentSpec,
    generate,
    compile_kernel,
)



def _set_printoptions():
    """Set the CuPy printoptions to show full matrices."""
    cp.set_printoptions(linewidth=200, threshold=sys.maxsize)


def test_add_kernel():
    """Compile and run a simple CUDA kernel."""
    module, add_kernel = compile_kernel(kernel_name="add")

    x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    y = cp.zeros((5, 5), dtype=cp.float32)
    add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
    cp.testing.assert_array_equal(x1 + x2, y)


def test_mma_m16_n8_k8_kernel():
    """Compile and run a GPU kernel that tests tensor core mma with shape m16_n8_k8."""
    module, mma_kernel = compile_kernel(
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
    module, mma_kernel = compile_kernel(
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
    # https://github.com/NVIDIA/cutlass/issues/1373#issuecomment-2121019973
    debug = False
    lineinfo = True

    N = 128
    C = 64

    # Arbitrary height and width work.
    H = 64
    W = 64

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

    WARPS = min(8, C8)
    THREADS = WARPS * 32

    BLOCK_C8 = WARPS
    BLOCK_Q = 16
    BLOCK_P = min(16, P)
    BLOCK_W = BLOCK_Q + S - 1
    BLOCK_C = BLOCK_C8 * 8

    BLOCK_C2 = BLOCK_C8 * 4

    blocks_n = N
    blocks_q = divup(Q, BLOCK_Q)
    blocks_p = divup(P, BLOCK_P)
    blocks_c8 = divup(C8, BLOCK_C8)
    BLOCKS = blocks_n * blocks_p * blocks_q * blocks_c8

    CHUNK_P = 1
    CHUNK_H = CHUNK_P + R - 1
    NUM_SMEM_INPUT_ROWS = CHUNK_P + CHUNK_H

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

    with GenDirectory() as include_dir:
        generate(
            [
                IndexSpec("BlockIdx", dict(n=N, c8=blocks_c8, p=blocks_p, q=blocks_q)),
                TensorSpec("Input", "const uint4", dict(n=N, y=H, x=W, c8=C8)),
                TensorSpec("Weights", "const uint4", dict(k=K, t=R, s=S)),
                TensorSpec(
                    "ConstSmemInput",
                    "const uint4",
                    dict(y=NUM_SMEM_INPUT_ROWS, x=BLOCK_W, c8=BLOCK_C8 + 1),
                ),
                TensorSpec(
                    "SmemInput",
                    "uint4",
                    dict(y=NUM_SMEM_INPUT_ROWS, x=BLOCK_W, c8=BLOCK_C8 + 1),
                ),
                IndexSpec("SmemInputLoadIdx", dict(c8=BLOCK_C8, repeat_x=2, x=16)),
                TensorSpec(
                    "ConstSmemWeights", "const uint4", dict(k=BLOCK_C, r=R, s=S)
                ),
                IndexSpec("SmemWeightsLoadIdx", dict(kd8=BLOCK_C8, repeat_k=4, km8=8)),
                TensorSpec(
                    "SmemOutput",
                    "__half2",
                    dict(p=CHUNK_P, q=BLOCK_Q, c2=BLOCK_C2 + 4),
                ),
                TensorSpec(
                    "ConstSmemOutput",
                    "const uint4",
                    dict(p=CHUNK_P, q=BLOCK_Q, c8=BLOCK_C8 + 1),
                ),
                IndexSpec("SmemOutputStoreIdx", dict(warp=WARPS, lane=32)),
                TensorSpec("Output", "uint4", dict(n=N, p=H, q=W, c8=C8)),
                IndexSpec("SmemOutputLoadIdx", dict(p=CHUNK_P, q=BLOCK_Q, c8=BLOCK_C8)),
                FragmentSpec("Acc", "MMA_M16_N8_F32_C", row="q", col="c"),
                ParamsSpec(
                    "Params",
                    dict(
                        N=N,
                        C=C,
                        H=H,
                        W=W,
                        R=R,
                        S=S,
                        C8=C8,
                        GROUP_WIDTH=group_width,
                        PADDING=PADDING,
                        WARPS=WARPS,
                        THREADS=THREADS,
                    ),
                ),
                ParamsSpec(
                    "Tiles",
                    dict(
                        BLOCK_Q=BLOCK_Q,
                        BLOCK_C8=BLOCK_C8,
                        BLOCK_W=BLOCK_Q + 2,
                        BLOCK_P=BLOCK_P,
                        CHUNK_H=CHUNK_H,
                        CHUNK_P=CHUNK_P,
                        NUM_SMEM_INPUT_ROWS=NUM_SMEM_INPUT_ROWS,
                    ),
                ),
            ],
            include_dir / "my_header.h",
        )

        module, conv_kernel = compile_kernel(
            kernel_name="conv_group_4_16w_4h_64c",
            source_file_name="conv_group_4",
            debug=debug,
            lineinfo=lineinfo,
            includes=[include_dir],
        )

        outputs = torch.zeros((N, C, H, W), device="cuda", dtype=torch.float16).to(
            memory_format=torch.channels_last
        )

        cp_outputs = cp.asarray(outputs)
        cp_inputs = cp.asarray(inputs)
        cp_weights = cp.asarray(conv.weight.detach().type(torch.float16))

        conv_kernel(
            (BLOCKS,),
            (THREADS,),
            (cp_outputs, cp_inputs, cp_weights),
        )

        diff = ref_outputs - outputs
        torch.testing.assert_close(ref_outputs, outputs)

        # _set_printoptions()
        # print(cp_loaded_input)


def test_memcpy_kernel():
    """This kernel achives 92% of peak DRAM memory bandwidth."""
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

    with GenDirectory() as include_dir:
        generate(
            [
                ParamsSpec(
                    "MyParams",
                    dict(
                        ITERS=ITERS,
                        BLOCK_X4=BLOCK_X4,
                        X=X,
                        THREADS=THREADS,
                    ),
                ),
            ],
            include_dir / "my_params.h",
        )

        module, memcpy_kernel = compile_kernel(
            kernel_name="memcpy_simple",
            source_file_name="memcpy_simple",
            debug=debug,
            lineinfo=lineinfo,
            includes=[include_dir],
        )

    inputs = torch.randn((N, C, H, W), device="cuda", dtype=torch.float32).to(
        memory_format=torch.channels_last
    )

    outputs = torch.zeros((N, C, H, W), device="cuda", dtype=torch.float32).to(
        memory_format=torch.channels_last
    )

    cp_inputs = cp.asarray(inputs)
    cp_outputs = cp.asarray(outputs)

    memcpy_kernel((BLOCKS,), (THREADS,), (cp_outputs, cp_inputs))

    diff = outputs - inputs
    assert torch.equal(outputs, inputs)


def test_row_memcpy_kernel():
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

    with GenDirectory() as include_dir:
        generate(
            [
                ParamsSpec("Block", dict(p=BLOCK_P, q=BLOCK_Q, c4=BLOCK_C4, padding=1)),
                IndexSpec(
                    "BlockIdx",
                    dict(n=BLOCKS_N, p=BLOCKS_P, q=BLOCKS_Q, c4=BLOCKS_C4),
                ),
                IndexSpec("InputIdx", dict(x=BLOCK_W, c4=BLOCK_C4)),
                TensorSpec("Input", "const float4", dict(n=N, y=H, x=W, c4=C4)),
                TensorSpec("Output", "float4", dict(n=N, p=H, q=W, c4=C4)),
                TensorSpec(
                    "SmemInput",
                    "float4",
                    dict(ping_pong=2, x=BLOCK_W, c4=BLOCK_C4 + 1),
                ),
                TensorSpec(
                    "ConstSmemInput",
                    "const float2",
                    dict(ping_pong=2, x=BLOCK_W, c4=BLOCK_C4 + 1, c2=2),
                ),
                IndexSpec("SmemInputLoadIdx", dict(c4=BLOCK_C4, q=BLOCK_Q, c2=2)),
                TensorSpec(
                    "SmemOutput",
                    "float2",
                    dict(q=BLOCK_Q, c4=BLOCK_C4 + 1, c2=2),
                ),
                TensorSpec(
                    "ConstSmemOutput",
                    "const float4",
                    dict(q=BLOCK_Q, c4=BLOCK_C4 + 1),
                ),
            ],
            include_dir / "parameters.h",
        )

        module, kernel = compile_kernel(
            kernel_name="row_memcpy",
            debug=debug,
            lineinfo=lineinfo,
            includes=[include_dir],
        )

    inputs = torch.randn((N, C, H, W), device="cuda", dtype=torch.float32).to(
        memory_format=torch.channels_last
    )

    outputs = torch.zeros((N, C, H, W), device="cuda", dtype=torch.float32).to(
        memory_format=torch.channels_last
    )

    cp_inputs = cp.asarray(inputs)
    cp_outputs = cp.asarray(outputs)

    kernel((BLOCKS,), (THREADS,), (cp_outputs, cp_inputs))
    assert torch.equal(outputs, inputs)
