"""This file contains an interface to the CUDA nvcc compiler.

This compiler is only used by certain unit tests which are disabled by
default. In general, Spio uses libnvrtc to compile CUDA code instead of
nvcc.

Therefore, nvcc is not requried for Spio to function correctly.
"""
from pathlib import Path
import os
import shutil
import subprocess

from .arch import sm_from_arch


def nvcc_full_path():
    """Return the path to nvcc or raise FileNotFoundError if not found.

    This function returns the value of the CUDACXX environment variable,
    if it is set. Else it returns "$CUDA_HOME / bin/ nvcc",  if the
    CUDA_HOME environment variable is set. Else it returns
    "/usr/local/cuda/bin/nvcc" if that file exists. Else it returns the
    result of using the "which" shell command to find "nvcc", if that
    returns a result. Else it raises a FileNotFoundError.
    """
    path = os.environ.get("CUDACXX")
    if path is not None:
        return path

    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home is not None:
        return str(Path(cuda_home) / "bin" / "nvcc")

    path = Path("/usr/local/cuda/bin/nvcc")
    if path.is_file():
        return str(path)

    path = shutil.which("nvcc")
    if path is not None:
        return path

    raise FileNotFoundError("Could not find nvcc.")


def compile_with_nvcc(
    sources,
    includes=None,
    run=False,
    cubin=False,
    compile_flag=False,
    arch=None,
    output_file=None,
    device_debug=False,
    lineinfo=False,
) -> int:
    """Deprecated.

    Use compile_with_nvrtc instead.
    """
    if includes is None:
        includes = []
    arch = sm_from_arch(arch)
    nvcc = nvcc_full_path()
    includes = [f"-I{path}" for path in includes]
    args = [nvcc] + includes
    if run:
        args.append("--run")
    if compile_flag:
        args.append("--compile")
    if cubin:
        args.append("--cubin")
    if arch is not None:
        args += ["-arch", arch]
    if output_file is not None:
        args += ["--output-file", output_file]
    if device_debug:
        args.append("-G")
    if lineinfo:
        args.append("-lineinfo")
    args += sources
    r = subprocess.run(args, check=True)
    r.check_returncode()
    return r
