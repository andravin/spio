import cupy as cp

from spio import spio_kernels_path, spio_cubins_path, compile

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
    assert cp.testing.numpy_cupy_allclose(x1 + x2, y)

