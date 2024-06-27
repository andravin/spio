import cupy as cp

from spio import spio_kernels_path, spio_cubins_path, compile, spio_include_path

ADA_ARCH = "sm_89"


def test_add_kernel():
    """Compile and run a simple CUDA kernel."""
    cuda_source_file = spio_kernels_path() / "add.cu"
    cubin_file = spio_cubins_path() / "add.cubin"
    compile(
        [cuda_source_file],
        compile=True,
        cubin=True,
        arch=ADA_ARCH,
        output_file=cubin_file,
    )

    module = cp.RawModule(path=str(cubin_file))
    add_kernel = module.get_function("my_add")

    x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    y = cp.zeros((5, 5), dtype=cp.float32)
    add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
    cp.testing.assert_array_equal(x1 + x2, y)


def test_mma_kernel():
    cuda_source_file = spio_kernels_path() / "mma.cu"
    cubin_file = spio_cubins_path() / "mma.cubin"
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
    mma_kernel = module.get_function("mma_test")

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
    mma_kernel((1, ), (32,), (C, A, B_trans))

    C_ref = cp.matmul(A.astype(cp.float32), B.astype(cp.float32))

    cp.testing.assert_array_equal(C_ref, C)

   # import sys
    # cp.set_printoptions(linewidth=200, threshold=sys.maxsize)

    # print("A:")
    # print(A)
    # print()

    # print("B:")
    # print(B)
    # print()

    # print("My kernel:")
    # print(C)
    # print()
    # print("cupy:")
    # print(C_ref)

 
