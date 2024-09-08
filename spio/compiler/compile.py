import os
import shutil
from pathlib import Path
import subprocess

# from ..cuda.nvrtc import Program, version as nvrtc_version
from ..cuda.nvrtc_ctypes import Program, version as nvrtc_version

CUDA_RUNTIME_INCLUDE_PATH = "/home/andrew/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/include"
USE_NVRTC = True

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


def compile(sources, **kwargs):
    if USE_NVRTC:
        return compile_with_nvrtc(sources, **kwargs)
    else:
        return compile_with_nvcc(sources, **kwargs)



def compile_with_nvrtc(
    sources,
    includes=[],
    run=False,
    cubin=False,
    compile=False,
    arch=None,
    output_file=None,
    device_debug=False,
    lineinfo=False,
): 
    arch = _sm_from_arch(arch)
    includes = includes + [CUDA_RUNTIME_INCLUDE_PATH]
    options = []
    if arch is not None:
        options.append(f"-arch={arch}")
    if device_debug:
        options.append("-G")
    if lineinfo:
        options.append("-lineinfo")
    options += [f"-I{path}" for path in includes]

    with open(sources[0], "r") as f:
        src = f.read()
    program = Program(src, "spio.cu")
    try:
        program.compile(options)
    except Exception as e:
        raise ValueError(f"Compilation error log: {program.log()}") from e
    cubin = program.cubin()    
    with open(output_file, "wb") as f:
        f.write(cubin)
    return 0


def compile_with_nvcc(
    sources,
    includes=[],
    run=False,
    cubin=False,
    compile=False,
    arch=None,
    output_file=None,
    device_debug=False,
    lineinfo=False,
):
    arch = _sm_from_arch(arch)
    nvcc = nvcc_full_path()
    includes = [f"-I{path}" for path in includes]
    args = [nvcc] + includes
    if run:
        args.append("--run")
    if compile:
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
    r = subprocess.run(args)
    r.check_returncode()
    return r


def _sm_from_arch(arch):
    if isinstance(arch, tuple):
        return f"sm_{arch[0]}{arch[1]}"
    else:
        return arch
