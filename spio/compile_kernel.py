from tempfile import NamedTemporaryFile
from typing import Tuple

import cupy as cp

from spio import compile, spio_cubins_path, spio_include_path, spio_kernels_path

ADA_ARCH = "sm_89"


def compile_kernel(
    kernel_name=None,
    source_file_name=None,
    includes=[],
    output_file=None,
    arch=ADA_ARCH,
    debug=False,
    lineinfo=False,
):
    cuda_source_file = spio_kernels_path() / source_file_name
    includes = includes + [spio_include_path()]
    return compile(
        [cuda_source_file],
        includes=includes,
        compile=True,
        cubin=True,
        arch=arch,
        output_file=output_file,
        device_debug=debug,
        lineinfo=lineinfo,
    )


def load_kernel(kernel_name: str, cubin_file_name: str = None):
    module = cp.RawModule(path=str(cubin_file_name))
    kernel = module.get_function(kernel_name)
    return (module, kernel)


def compile_and_load_kernel(
    kernel_name=None,
    source_file_name=None,
    includes=[],
    output_file=None,
    arch=ADA_ARCH,
    debug=False,
    lineinfo=False,
):
    if source_file_name is None:
        source_file_name = f"{kernel_name}.cu"
    with NamedTemporaryFile() as cubin_file:
        r = compile_kernel(
            kernel_name=kernel_name,
            source_file_name=source_file_name,
            debug=debug,
            lineinfo=lineinfo,
            includes=includes,
            output_file=cubin_file.name,
            arch=arch,
        )
        return load_kernel(kernel_name, cubin_file.name)
