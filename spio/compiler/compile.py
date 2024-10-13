import importlib.resources
from importlib.abc import Traversable

from ..cuda.nvrtc_ctypes import Program, version as nvrtc_version

from .arch import sm_from_arch


def _find_cuda_runtime_include_dir() -> str:
    return str(importlib.resources.files("nvidia.cuda_runtime").joinpath("include"))


CUDA_RUNTIME_INCLUDE_PATH = _find_cuda_runtime_include_dir()


def compile(
    src_file:Traversable,
    includes=[],
    arch=None,
    device_debug=False,
    lineinfo=False,
    header_dict=None,
):
    arch = sm_from_arch(arch)
    includes = includes + [CUDA_RUNTIME_INCLUDE_PATH]
    options = []
    if arch is not None:
        options.append(f"-arch={arch}")
    if device_debug:
        options.append("-G")
    if lineinfo:
        options.append("-lineinfo")
    options += [f"-I{path}" for path in includes]

    if header_dict is not None:
        headers = list(header_dict.values())
        include_names = list(header_dict.keys())
    else:
        headers = []
        include_names = []

    src = src_file.read_text()
    program = Program(src, "spio.cu", headers=headers, include_names=include_names)
    try:
        program.compile(options)
    except Exception as e:
        raise ValueError(f"Compilation error log: {program.log()}") from e
    return program.cubin()
