from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from contextvars import ContextVar
from functools import partial
import os

from .compile_kernel import compile_kernel

TRUTHS = ["true", "1", "yes", "y", "t"]

default_lineinfo = os.environ.get("SPIO_LINEINFO", "False").lower() in TRUTHS
default_debug = os.environ.get("SPIO_DEBUG", "False").lower() in TRUTHS
workers = int(os.environ.get("SPIO_WORKERS", "4"))

lineinfo = ContextVar("lineinfo", default=default_lineinfo)
debug = ContextVar("debug", default=default_debug)

pool = Pool(workers)


def compile_kernel_configs(kernel_cls, params, configs=None, **kernel_kwargs):
    """Compile multiple kernel configurations for a given kernel class.

    Use the given list of kernel configuratio objects, or enumerate all valid configurations if none are given.
    """
    if configs is None:
        configs = list(kernel_cls.configs(params))
    kernels = [kernel_cls(params, config=config, **kernel_kwargs) for config in configs]
    compiler_args = [kernel.compiler_args for kernel in kernels]
    _compile_kernels(compiler_args)
    return kernels


def _compile_kernels(compiler_args):
    """Compile multiple kernels in parallel."""
    ck_with_args = partial(compile_kernel, lineinfo=lineinfo.get(), debug=debug.get())
    async_result = pool.starmap_async(ck_with_args, compiler_args)
    return async_result.get()
