"""Function for compling CUDA source code into binary."""

from typing import List, Dict, Tuple

from importlib_resources import files as importlib_resources_files
from importlib_resources.abc import Traversable

from ..cuda.nvrtc_ctypes import Program

from .arch import sm_from_arch


def _find_cuda_runtime_include_dir() -> str:
    return str(importlib_resources_files("nvidia.cuda_runtime").joinpath("include"))


CUDA_RUNTIME_INCLUDE_PATH = _find_cuda_runtime_include_dir()


def compile_cuda(
    src_file: Traversable,
    includes: List[str] = None,
    arch: Tuple[int, int] = None,
    device_debug: bool = False,
    lineinfo: bool = False,
    header_dict: Dict[str, str] = None,
    max_registers: int = None,
    default_device: bool = True,
) -> bytes:
    """Compile CUDA source code and return the resulting cubin.

    Args:
        src_file: The CUDA source file to compile.
        includes: Additional include directories.
        arch: The GPU architecture to target.
        device_debug: Whether to include debugging information.
        lineinfo: Whether to include line information.
        header_dict: A dictionary of header file names and contents.
        max_registers: Maximum number of registers to use for the kernel.
    """
    arch = sm_from_arch(arch)
    if includes is None:
        includes = []
    includes = includes + [CUDA_RUNTIME_INCLUDE_PATH]
    options = []
    if arch is not None:
        options.append(f"-arch={arch}")
    if device_debug:
        options.append("-G")
    if lineinfo:
        options.append("-lineinfo")
    if max_registers is not None:
        options.append(f"-maxrregcount={max_registers}")
    
    # --extended-lambda or --expt-extended-lambda are supposed to allow adding __device__ labels
    # to lambda expressions, but it did not work for me. So I'm using -default-device instead.
    if default_device:
        options.append("-default-device")
    
    options += [f"-I{path}" for path in includes]

    if header_dict is not None:
        headers = list(header_dict.values())
        include_names = list(header_dict.keys())
    else:
        headers = []
        include_names = []

    src = src_file.read_text()
    program = Program(src, src_file.name, headers=headers, include_names=include_names)
    try:
        program.compile(options)
    except Exception as e:
        raise ValueError(f"Compilation error log: {program.log()}") from e
    return program.cubin()
