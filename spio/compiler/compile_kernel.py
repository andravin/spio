from tempfile import NamedTemporaryFile
from typing import Tuple

import cupy as cp
import torch
from .paths import (
    spio_cubins_path,
    spio_include_path,
    spio_kernels_path,
    spio_test_kernels_path,
)

from .compile import compile


def compile_kernel(
    kernel_name=None,
    source_file_name=None,
    includes=[],
    output_file=None,
    arch=None,
    debug=False,
    lineinfo=False,
    test_kernel=False,
):
    assert arch is not None, "Must specify GPU architecture for kernel compilation."
    if arch[0] < 8:
        raise ValueError(
            "Minimum supported GPU compute capability is sm_80 (Ampere or newer)."
        )
    kernel_path = spio_test_kernels_path() if test_kernel else spio_kernels_path()
    cuda_source_file = kernel_path / source_file_name
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
    device=None,
    debug=False,
    lineinfo=False,
    test_kernel=False,
):
    arch = torch.cuda.get_device_capability(device)
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
            test_kernel=test_kernel,
        )
        return load_kernel(kernel_name, cubin_file.name)
