"""This file contains an interface to the CUDA nvcc compiler.

This compiler is only used by certain unit tests which are disabled by
default. In general, Spio uses libnvrtc to compile CUDA code instead of
nvcc.

Therefore, nvcc is not requried for Spio to function correctly.
"""

import subprocess

from .cuda_paths import nvcc_full_path
from .arch import sm_from_arch


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
    pre_includes=None,
    run_args=None,
) -> int:
    """Compile CUDA source files with nvcc.

    Kernel compilation should use compile_with_nvrtc instead.

    This function is used for C++ unit tests.
    """
    if includes is None:
        includes = []
    arch = sm_from_arch(arch)
    nvcc = nvcc_full_path()
    includes = [f"-I{path}" for path in includes]
    args = [nvcc] + includes
    if pre_includes is not None:
        pre_include_str = ",".join(pre_includes)
        args += [f"--pre-include={pre_include_str}"]
    if run:
        args.append("--run")
    if run_args is not None:
        run_args_str = ",".join(run_args)
        args += [f"--run-args={run_args_str}"]
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
