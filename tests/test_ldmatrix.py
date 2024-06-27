import cupy as cp

from spio import compile_test_kernel


def test_ldmatrix_kernel():
    """Compile and run an ldmatrix test kernel."""
    module, ldmatrix_kernel = compile_test_kernel(kernel_name="ldmatrix")

    a = cp.arange(8 * 8, dtype=cp.float16).reshape(8, 8)
    b = cp.zeros((64,), dtype=cp.float16)

    ldmatrix_kernel((1,), (32,), (b, a))

    for lane in range(32):
        row = lane / 4
        col = (lane % 4) * 2
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
        row = lane / 4
        col = (lane % 4) * 2
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
        row = lane / 4
        col = (lane % 4) * 2
        for fragment in range(4):
            idx = lane * 2 + fragment * 64
            assert b[idx + 0] == a[fragment * 8 + row, col]
            assert b[idx + 1] == a[fragment * 8 + row, col + 1]
