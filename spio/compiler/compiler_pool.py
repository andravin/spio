from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from contextvars import ContextVar
from functools import partial
import os

from .compile_kernel import compile_kernel

TRUTHS = ['true', '1', 'yes', 'y', 't']

default_lineinfo = os.environ.get("spio_lineinfo", 'False').lower() in TRUTHS
default_debug = os.environ.get("spio_debug", 'False').lower() in TRUTHS
workers = int(os.environ.get("spio_workers", '4'))

lineinfo = ContextVar("lineinfo", default=default_lineinfo)
debug = ContextVar("debug", default=default_debug)

pool = Pool(workers)

def compile_kernels(compiler_args):
    ck_with_args = partial(compile_kernel, lineinfo=lineinfo.get(), debug=debug.get())
    async_result = pool.starmap_async(ck_with_args, compiler_args)
    return async_result.get()
