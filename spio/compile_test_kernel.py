import cupy as cp

from spio import compile, spio_cubins_path, spio_include_path, spio_kernels_path

ADA_ARCH = "sm_89"


def compile_test_kernel(kernel_name=None, source_file_name=None):
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
