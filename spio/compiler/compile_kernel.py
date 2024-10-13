from typing import Tuple
import importlib.resources

import torch

from .. import primary_context_guard
from ..cuda.driver import Module, Function
from .compile import compile


def compile_kernel(
    kernel_name=None,
    source_file_name=None,
    header_dict=None,
    src_module="spio.src",
    includes_module="spio.include",
    arch=None,
    debug=False,
    lineinfo=False,
):
    assert arch is not None, "Must specify GPU architecture for kernel compilation."
    if arch[0] < 8:
        raise ValueError(
            "Minimum supported GPU compute capability is sm_80 (Ampere or newer)."
        )
    cuda_source_file = importlib.resources.files(src_module).joinpath(source_file_name)
    includes_dir = str(importlib.resources.files(includes_module))
    return compile(
        cuda_source_file,
        includes=[includes_dir],
        arch=arch,
        device_debug=debug,
        lineinfo=lineinfo,
        header_dict=header_dict,
    )


def load_kernel(
    kernel_name: str, cubin: str = None, device_ordinal: int = 0
) -> Tuple[Module, Function]:
    primary_context_guard.set_device(device_ordinal)
    module = Module()
    module.load_data(cubin)
    function = module.get_function(kernel_name)
    return (module, function)


def compile_and_load_kernel(
    kernel_name=None,
    source_file_name=None,
    header_dict=None,
    src_module="spio.src",
    includes_module="spio.include",
    device_ordinal=0,
    debug=False,
    lineinfo=False,
):
    arch = torch.cuda.get_device_capability(device_ordinal)
    if source_file_name is None:
        source_file_name = f"{kernel_name}.cu"
    cubin = compile_kernel(
        kernel_name=kernel_name,
        source_file_name=source_file_name,
        src_module=src_module,
        includes_module=includes_module,
        header_dict=header_dict,
        debug=debug,
        lineinfo=lineinfo,
        arch=arch,
    )
    return load_kernel(kernel_name, cubin, device_ordinal=device_ordinal)
