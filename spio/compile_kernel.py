from tempfile import NamedTemporaryFile

import cupy as cp

from spio import compile, spio_cubins_path, spio_include_path, spio_kernels_path

ADA_ARCH = "sm_89"


def compile_kernel(
    kernel_name=None,
    source_file_name=None,
    debug=False,
    lineinfo=False,
    includes=[],
    arch=ADA_ARCH,
):
    """Compile the kernel with the given name.

    By default, the kernel must be in a file called {kernel_name}.cu
    located in the spio_kernels_path() folder.

    If the source_file_name parameter is set, then the
    kernel is loaded from the file {source_file_name}.cu
    instead.

    Returns the module and RawKernel (both CuPy objects).
    """
    if source_file_name is None:
        cuda_source_file = spio_kernels_path() / f"{kernel_name}.cu"
    else:
        cuda_source_file = spio_kernels_path() / f"{source_file_name}.cu"

    with NamedTemporaryFile(
        suffix=".cubin", prefix=f"spio_{kernel_name}_"
    ) as cubin_file:
        includes = includes + [spio_include_path()]
        compile(
            [cuda_source_file],
            includes=includes,
            compile=True,
            cubin=True,
            arch=arch,
            output_file=cubin_file.name,
            device_debug=debug,
            lineinfo=lineinfo,
        )
        module = cp.RawModule(path=str(cubin_file.name))
        kernel = module.get_function(kernel_name)
    return (module, kernel)
