"""This file contains functions to find CUDA toolkit binaries.

Generally the CUDA toolkit is not required for Spio, but its presence
enable additional testing and diagnostics.
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil


def _cuda_path() -> Path | None:
    """Return the path of the CUDA toolkit installation or None if not found."""
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home is not None:
        return Path(cuda_home)

    path = Path("/usr/local/cuda")
    if path.is_dir():
        return path

    return None


def has_cuda_toolkit() -> bool:
    """Return True if the CUDA toolkit is installed, False otherwise."""
    try:
        nvcc_full_path()
        return True
    except FileNotFoundError:
        return False


def nvcc_full_path() -> str:
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

    cuda_path = _cuda_path()
    if cuda_path is not None:
        nvcc_path = cuda_path / "bin" / "nvcc"
        if not nvcc_path.is_file():
            raise FileNotFoundError(
                f"Could not find nvcc at expected location: {nvcc_path}"
            )
        return str(nvcc_path)

    path = shutil.which("nvcc")
    if path is not None:
        return path

    raise FileNotFoundError("Could not find nvcc.")


def nvdisasm_full_path() -> str:
    """Return the path to nvdisasm or raise FileNotFoundError if not found."""
    cuda_path = _cuda_path()
    if cuda_path is None:
        raise FileNotFoundError(
            "Could not find nvdisasm because the CUDA toolkit is not installed."
        )
    nvdisasm_path = cuda_path / "bin" / "nvdisasm"
    if not nvdisasm_path.is_file():
        raise FileNotFoundError(
            f"Could not find nvdisasm at expected location: {nvdisasm_path}"
        )
    return str(nvdisasm_path)
