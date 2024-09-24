from tempfile import NamedTemporaryFile
from typing import Tuple

import torch

from ..cuda import driver, primary_context_guard
from .paths import (
    spio_src_path,
    spio_test_src_path,
    spio_include_path,
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
    if test_kernel:
        cuda_source_file = spio_test_src_path(source_file_name)
    else:
        cuda_source_file = spio_src_path(source_file_name)
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


def load_kernel(
    kernel_name: str, cubin_file_name: str = None, device_ordinal: int = 0
) -> Tuple[driver.Module, driver.Function]:
    primary_context_guard.set_device(device_ordinal)
    module = driver.Module()
    module.load(str(cubin_file_name))
    function = module.get_function(kernel_name)
    return (module, function)


def compile_and_load_kernel(
    kernel_name=None,
    source_file_name=None,
    includes=[],
    output_file=None,
    device_ordinal=0,
    debug=False,
    lineinfo=False,
    test_kernel=False,
):
    arch = torch.cuda.get_device_capability(device_ordinal)
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
        return load_kernel(kernel_name, cubin_file.name, device_ordinal=device_ordinal)
