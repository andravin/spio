from dataclasses import dataclass
import logging

from .benchmark import benchmark
from .benchmark_formatter import BenchmarkResultCompactFormat, BenchmarkResultFullFormat
from .kernel_util import get_first_device_in_args
from .benchmark_result import BenchmarkResult
from .benchmark_logger import benchmark_logger

@dataclass(frozen=True)
class _KernelKey:
    device_ordinal: int
    params: object


class KernelCache:
    def __init__(self):
        self._cache = {}

    def get(self, kernel_cls, params, args, config=None, **kernel_kwargs):
        device = get_first_device_in_args(args)
        if config is None:
            key = _KernelKey(device_ordinal=device.index, params=params)
            best_kernel = self._cache.get(key)
            if best_kernel is None:
                best_result, best_kernel, results = benchmark(
                    kernel_cls, params, [args], device=device, **kernel_kwargs
                )
                self._cache[key] = best_kernel
                benchmark_logger.log_results(results)
                benchmark_logger.log_best(best_result)
            return best_kernel
        else:
            return kernel_cls(params, config=config, **kernel_kwargs)
