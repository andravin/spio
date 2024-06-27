"""Tests that compile and run CUDA kernels that use the ldmatrix instructions.

The ldmatrix instructions load matrix fragments from shared memory into registers.
The matrix fragments can be used with the tensor core matrix multiply instructions in mma.h.
"""

import cupy as cp

from spio import compile_test_kernel


def _row(lane: int) -> int:
    """Return the row loaded by the register in the given lane."""
    return lane / 4


def _col(lane: int) -> int:
    """Return the (first) column loaded by the register in the given lane."""
    return (lane % 4) * 2


def test_ldmatrix_kernel():
    """Compile and run an ldmatrix test kernel."""
    module, ldmatrix_kernel = compile_test_kernel(kernel_name="ldmatrix")

    a = cp.arange(8 * 8, dtype=cp.float16).reshape(8, 8)
    b = cp.zeros((64,), dtype=cp.float16)

    ldmatrix_kernel((1,), (32,), (b, a))

    for lane in range(32):
        row = _row(lane)
        col = _col(lane)
        idx = lane * 2
        assert b[idx + 0] == a[row, col]
        assert b[idx + 1] == a[row, col + 1]


def test_ldmatrix_x2_kernel():
    """Compile and run an ldmatrix_x2 test kernel."""
    module, ldmatrix_x2_kernel = compile_test_kernel(
        kernel_name="ldmatrix_x2", source_file_name="ldmatrix"
    )

    a = cp.arange(16 * 8, dtype=cp.float16).reshape(16, 8)
    b = cp.zeros((16 * 8,), dtype=cp.float16)

    ldmatrix_x2_kernel((1,), (32,), (b, a))

    for lane in range(32):
        row = _row(lane)
        col = _col(lane)
        for fragment in range(2):
            idx = lane * 2 + fragment * 64
            assert b[idx + 0] == a[fragment * 8 + row, col]
            assert b[idx + 1] == a[fragment * 8 + row, col + 1]


def test_ldmatrix_x4_kernel():
    """Compile and run an ldmatrix_x4 test kernel."""
    module, ldmatrix_x4_kernel = compile_test_kernel(
        kernel_name="ldmatrix_x4", source_file_name="ldmatrix"
    )

    a = cp.arange(64 * 8, dtype=cp.float16).reshape(64, 8)
    b = cp.zeros((64 * 8,), dtype=cp.float16)

    ldmatrix_x4_kernel((1,), (32,), (b, a))

    for lane in range(32):
        row = _row(lane)
        col = _col(lane)
        for fragment in range(4):
            idx = lane * 2 + fragment * 64
            assert b[idx + 0] == a[fragment * 8 + row, col]
            assert b[idx + 1] == a[fragment * 8 + row, col + 1]
