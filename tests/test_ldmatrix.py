"""Tests that compile and run CUDA kernels that use the ldmatrix instructions.

The ldmatrix instructions load matrix fragments from shared memory into registers.
The matrix fragments can be used with the tensor core matrix multiply instructions in mma.cuh.
"""

import torch

from spio.compiler import compile_and_load_kernel


def _row(lane: int) -> int:
    """Return the row loaded by the register in the given lane."""
    return lane // 4


def _col(lane: int) -> int:
    """Return the (first) column loaded by the register in the given lane."""
    return (lane % 4) * 2


def test_ldmatrix_x1_kernel():
    """Compile and run an ldmatrix test kernel."""
    _, ldmatrix_kernel = compile_and_load_kernel(
        kernel_name="ldmatrix_x1", source_file_name="ldmatrix.cu", src_module="spio.src_tests"
    )

    a = torch.arange(8 * 8, dtype=torch.float16, device="cuda").reshape(8, 8)
    b = torch.zeros((64,), dtype=torch.float16, device="cuda")

    ldmatrix_kernel.launch((1, 1, 1), (32, 1, 1), (b, a))

    for lane in range(32):
        row = _row(lane)
        col = _col(lane)
        idx = lane * 2
        assert b[idx + 0] == a[row, col]
        assert b[idx + 1] == a[row, col + 1]


def test_ldmatrix_x2_kernel():
    """Compile and run an ldmatrix_x2 test kernel."""
    _, ldmatrix_x2_kernel = compile_and_load_kernel(
        kernel_name="ldmatrix_x2", source_file_name="ldmatrix.cu", src_module="spio.src_tests"
    )

    a = torch.arange(16 * 8, dtype=torch.float16, device="cuda").reshape(16, 8)
    b = torch.zeros((16 * 8,), dtype=torch.float16, device="cuda")

    ldmatrix_x2_kernel.launch((1, 1, 1), (32, 1, 1), (b, a))

    for lane in range(32):
        row = _row(lane)
        col = _col(lane)
        for fragment in range(2):
            idx = lane * 2 + fragment * 64
            assert b[idx + 0] == a[fragment * 8 + row, col]
            assert b[idx + 1] == a[fragment * 8 + row, col + 1]


def test_ldmatrix_x4_kernel():
    """Compile and run an ldmatrix_x4 test kernel."""
    _, ldmatrix_x4_kernel = compile_and_load_kernel(
        kernel_name="ldmatrix_x4", source_file_name="ldmatrix.cu", src_module="spio.src_tests"
    )

    a = torch.arange(64 * 8, dtype=torch.float16, device="cuda").reshape(64, 8)
    b = torch.zeros((64 * 8,), dtype=torch.float16, device="cuda")

    ldmatrix_x4_kernel.launch((1, 1, 1), (32, 1, 1), (b, a))

    for lane in range(32):
        row = _row(lane)
        col = _col(lane)
        for fragment in range(4):
            idx = lane * 2 + fragment * 64
            assert b[idx + 0] == a[fragment * 8 + row, col]
            assert b[idx + 1] == a[fragment * 8 + row, col + 1]


def test_ldmatrix_x1_trans_kernel():
    """Compile and run an ldmatrix_x1_trans test kernel."""
    _, ldmatrix_x1_trans_kernel = compile_and_load_kernel(
        kernel_name="ldmatrix_x1_trans",
        source_file_name="ldmatrix.cu",
        src_module="spio.src_tests",
    )

    a = torch.arange(8 * 8, dtype=torch.float16, device="cuda").reshape(8, 8)
    b = torch.zeros((64,), dtype=torch.float16, device="cuda")

    ldmatrix_x1_trans_kernel.launch((1, 1, 1), (32, 1, 1), (b, a))

    for lane in range(32):
        row = _row(lane)
        col = _col(lane)
        idx = lane * 2
        assert b[idx + 0] == a[col, row]
        assert b[idx + 1] == a[col + 1, row]


def test_ldmatrix_x2_trans_kernel():
    """Compile and run an ldmatrix_x2 test kernel."""
    _, ldmatrix_x2_trans_kernel = compile_and_load_kernel(
        kernel_name="ldmatrix_x2_trans",
        source_file_name="ldmatrix.cu",
        src_module="spio.src_tests",
    )

    a = torch.arange(16 * 8, dtype=torch.float16, device="cuda").reshape(16, 8)
    b = torch.zeros((16 * 8,), dtype=torch.float16, device="cuda")

    ldmatrix_x2_trans_kernel.launch((1, 1, 1), (32, 1, 1), (b, a))

    for lane in range(32):
        row = _row(lane)
        col = _col(lane)
        for fragment in range(2):
            idx = lane * 2 + fragment * 64
            assert b[idx + 0] == a[fragment * 8 + col, row]
            assert b[idx + 1] == a[fragment * 8 + col + 1, row]


def test_ldmatrix_x4_trans_kernel():
    """Compile and run an ldmatrix_x4_trans test kernel."""
    _, ldmatrix_x4_trans_kernel = compile_and_load_kernel(
        kernel_name="ldmatrix_x4_trans",
        source_file_name="ldmatrix.cu",
        src_module="spio.src_tests",
    )

    a = torch.arange(64 * 8, dtype=torch.float16, device="cuda").reshape(64, 8)
    b = torch.zeros((64 * 8,), dtype=torch.float16, device="cuda")

    ldmatrix_x4_trans_kernel.launch((1, 1, 1), (32, 1, 1), (b, a))

    for lane in range(32):
        row = _row(lane)
        col = _col(lane)
        for fragment in range(4):
            idx = lane * 2 + fragment * 64
            assert b[idx + 0] == a[fragment * 8 + col, row]
            assert b[idx + 1] == a[fragment * 8 + col + 1, row]
