"""Compile kernels in parallel using a process pool."""

from multiprocessing import Pool
from contextvars import ContextVar
from functools import partial
import os
import signal
from typing import List, Any, Tuple, TYPE_CHECKING

from ..cuda.driver import DeviceAttributes

from .compile_kernel import compile_kernel

if TYPE_CHECKING:
    from ..kernels import Kernel, KernelFactory, Params


TRUTHS = ["true", "1", "yes", "y", "t"]
LONG_TIMEOUT = 999
DEFAULT_WORKERS = 1

default_lineinfo = os.environ.get("SPIO_LINEINFO", "False").lower() in TRUTHS
default_debug = os.environ.get("SPIO_DEBUG", "False").lower() in TRUTHS
workers = int(os.environ.get("SPIO_WORKERS", f"{DEFAULT_WORKERS}"))

lineinfo = ContextVar("lineinfo", default=default_lineinfo)
debug = ContextVar("debug", default=default_debug)

# Let the pool worker processes ignore SIGINT.
# Reference:
# https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
pool = Pool(workers)
signal.signal(signal.SIGINT, old_handler)


def compile_kernel_configs(
    kernel_factory: "KernelFactory",
    params: "Params",
    configs: List[Any] = None,
    device_attr: DeviceAttributes = None,
    **kernel_kwargs,
) -> List["Kernel"]:
    """Compile multiple kernel configurations in parallel.

    Use the given list of kernel configuration objects, or enumerate all valid configurations
    if none are given.

    The returned Kernnel objects will have their cubin attribute set to the compiled binary.
    You must call the kernel's load method to load the binary into a device
    after which you can call the kernel's launch method to execute it.

    Args:
        kernel_factory (KernelFactory): The factory to create kernels.
        params (Params): The parameters for the kernels.
        configs (List[Any], optional): List of kernel configuration objects.
        arch (Tuple[int, int]): Compute capability to compile for.
        **kernel_kwargs: Additional keyword arguments for kernel creation.

    Returns:
        List[Kernel]: List of compiled kernel objects.
    """
    if configs is None:
        configs = list(kernel_factory.configs(params, device_attr, **kernel_kwargs))
    kernels = [
        kernel_factory.make_kernel(
            params, config=config, device_attr=device_attr, **kernel_kwargs
        )
        for config in configs
    ]
    compile_kernels(kernels, arch=device_attr.compute_capability)
    return kernels


def compile_kernels(kernels: List["Kernel"], arch: Tuple[int, int] = None) -> None:
    """Compile multiple kernel objects in parallel.

    Args:
        kernels (List[Kernel]): List of kernel objects to compile.
        arch (Tuple[int, int]): Compute capability to compile for.
    """
    compiler_args = [kernel.compiler_args for kernel in kernels]
    cubins = _compile_kernels(compiler_args, arch=arch)
    for kernel, cubin in zip(kernels, cubins):
        kernel.cubin = cubin


def _compile_kernels(compiler_args, arch: Tuple[int, int] = None) -> List[bytes]:
    """Compile multiple kernels in parallel."""
    ck_with_args = partial(
        compile_kernel, arch=arch, lineinfo=lineinfo.get(), debug=debug.get()
    )
    try:
        async_result = pool.starmap_async(ck_with_args, compiler_args)
        res = async_result.get(LONG_TIMEOUT)
    except KeyboardInterrupt as e:
        pool.terminate()
        pool.join()
        raise e
    except Exception as e:
        raise ValueError("Error compiling kernels") from e
    return res
