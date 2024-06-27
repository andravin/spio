"""Unit tests that compile and run cuda kernels and check the output.

These tests exercise the spio CUDA device functions, such as tensor
core and ldmatrix intrinsics.
"""

import cupy as cp

from spio import spio_kernels_path, spio_cubins_path, compile, spio_include_path

ADA_ARCH = "sm_89"


def _set_printoptions():
    """Set the CuPy printoptions to show full matrices."""
    cp.set_printoptions(linewidth=200, threshold=sys.maxsize)


def _compile_test_kernel(kernel_name=None, source_file_name=None):
    """Compile the kernel for the test with the given name.

    The kernel must be in a file called {kernel_name}.cu
    located in the spio_kernels_path() folder.

    The kernel function must be called {kernel_name}_test

    Returns the module and RawKernel (both CuPy objects).
    """
    if source_file_name is None:
        cuda_source_file = spio_kernels_path() / f"{kernel_name}.cu"
    else:
        cuda_source_file = spio_kernels_path() / f"{source_file_name}.cu"
    cubin_file = spio_cubins_path() / f"{kernel_name}.cubin"
    include_path = spio_include_path()

    compile(
        [cuda_source_file],
        includes=[include_path],
        compile=True,
        cubin=True,
        arch=ADA_ARCH,
        output_file=cubin_file,
    )

    module = cp.RawModule(path=str(cubin_file))
    mma_kernel = module.get_function(f"{kernel_name}_test")
    return (module, mma_kernel)


def test_add_kernel():
    """Compile and run a simple CUDA kernel."""
    module, add_kernel = _compile_test_kernel(kernel_name="add")

    x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    y = cp.zeros((5, 5), dtype=cp.float32)
    add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
    cp.testing.assert_array_equal(x1 + x2, y)


def test_mma_kernel():
    """Compile and run a tensor core test kernel."""
    module, mma_kernel = _compile_test_kernel(kernel_name="mma")

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


def test_ldmatrix_kernel():
    """Compile and run an ldmatrix test kernel."""
    module, ldmatrix_kernel = _compile_test_kernel(kernel_name="ldmatrix")

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
    module, ldmatrix_x2_kernel = _compile_test_kernel(
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
    module, ldmatrix_x4_kernel = _compile_test_kernel(
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
