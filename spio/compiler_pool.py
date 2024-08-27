from multiprocessing import Pool
from multiprocessing.pool import AsyncResult

from .compile_kernel import compile_kernel

# TODO make this configurable
WORKERS = 32

pool = Pool(WORKERS)


def compile_kernels(compiler_args):
    async_result = pool.starmap_async(compile_kernel, compiler_args)
    return async_result.get()
