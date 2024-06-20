import os
import shutil
from pathlib import Path
import subprocess

def nvcc_full_path():
    """Return the path to nvcc or raise FileNotFoundError if not found.

    This function returns the value of the CUDACXX environment variable, if it is set.
    Else it returns "$CUDA_HOME / bin/ nvcc",  if the CUDA_HOME environment variable is set.
    Else it returns "/usr/local/cuda/bin/nvcc" if that file exists.
    Else it returns the result of using the "which" shell command to find "nvcc", if that returns a result.
    Else it raises a FileNotFoundError.
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


def compile(sources=[], includes=[], run=False, cubin=False, arch=None):
    nvcc = nvcc_full_path()
    includes = [f"-I{path}" for path in includes]   
    args = [nvcc] + includes
    if run:
        args.append("--run")
    if cubin:
        args.append("--cubin")
    if arch is not None:
        args += ["-arch", arch]
    args += sources
    r = subprocess.run(args)
    r.check_returncode()
    return r
