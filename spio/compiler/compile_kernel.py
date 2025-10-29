"""Functions for compiling and loading CUDA kernels."""

from typing import Tuple, Dict

import torch
from importlib_resources import files as importlib_resources_files

from .. import primary_context_guard
from ..cuda.driver import Module, Function
from .compile import compile_cuda


def compile_kernel(
    source_file_name: str = None,
    header_dict: Dict[str, str] = None,
    src_module: str = "spio.src",
    includes_module: str = "spio.include",
    arch: Tuple[int, int] = None,
    debug: bool = False,
    lineinfo: bool = False,
    max_registers: int = None,
):
    """Compile a CUDA kernel from a source file and return the cubin.

    Args:
        source_file_name (str): Name of the source file containing the kernel.
        header_dict (dict): Dictionary of header file names and sources.
        src_module (str): Module containing the source file.
        includes_module (str): Module containing the include files.
        arch (tuple): GPU architecture (e.g., (8, 0) for sm_80).
        debug (bool): Whether to include debug information.
        lineinfo (bool): Whether to include line information.
        max_registers (int): Maximum number of registers to use for the kernel.
    """
    assert arch is not None, "Must specify GPU architecture for kernel compilation."
    if arch[0] < 8:
        raise ValueError(
            "Minimum supported GPU compute capability is sm_80 (Ampere or newer)."
        )
    cuda_source_file = importlib_resources_files(src_module).joinpath(source_file_name)
    includes_dir = str(importlib_resources_files(includes_module))
    return compile_cuda(
        cuda_source_file,
        includes=[includes_dir],
        arch=arch,
        device_debug=debug,
        lineinfo=lineinfo,
        header_dict=header_dict,
        max_registers=max_registers,
    )


def load_kernel(
    kernel_name: str, cubin: str = None, device_ordinal: int = 0
) -> Tuple[Module, Function]:
    """Load a compiled CUDA kernel from cubin.

    Args:
        kernel_name (str): Name of the kernel to load.
        cubin (str): The CUDA binary.
        device_ordinal (int): The device number on which to load the kernel.
    """
    primary_context_guard.set_device(device_ordinal)
    module = Module()
    module.load_data(cubin)
    function = module.get_function(kernel_name)
    return (module, function)


def compile_and_load_kernel(
    kernel_name: str = None,
    source_file_name: str = None,
    header_dict: Dict[str, str] = None,
    src_module: str = "spio.src",
    includes_module: str = "spio.include",
    device_ordinal: int = 0,
    debug: bool = False,
    lineinfo: bool = False,
    max_registers: int = None,
):
    """Compile and load a CUDA kernel."""
    arch = torch.cuda.get_device_capability(device_ordinal)
    if source_file_name is None:
        source_file_name = f"{kernel_name}.cu"
    cubin = compile_kernel(
        source_file_name=source_file_name,
        src_module=src_module,
        includes_module=includes_module,
        header_dict=header_dict,
        debug=debug,
        lineinfo=lineinfo,
        arch=arch,
        max_registers=max_registers,
    )
    return load_kernel(kernel_name, cubin, device_ordinal=device_ordinal)
