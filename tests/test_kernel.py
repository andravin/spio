"""Unit tests that compile and run cuda kernels and check the output.

These tests exercise the spio CUDA device functions, such as tensor
core and ldmatrix intrinsics.
"""

import cupy as cp

from spio import compile_test_kernel

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


def test_mma_kernel():
    """Compile and run a tensor core test kernel."""
    module, mma_kernel = compile_test_kernel(kernel_name="mma")

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