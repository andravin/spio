from dataclasses import dataclass
import logging


from .benchmark_kernel import benchmark_kernel
from ..log import (
    BenchmarkResultCompactFormat,
    BenchmarkResultFullFormat,
    BenchmarkResult,
    benchmark_logger,
)
from .kernel_util import get_first_device_in_args
from .performance_model_cache import PerformanceModelCache

@dataclass(frozen=True)
class _KernelKey:
    device_ordinal: int
    params: object

perf_model_cache = PerformanceModelCache()

class KernelCache:
    def __init__(self):
        self._cache = {}

    def get(self, kernel_cls, params, args, config=None, **kernel_kwargs):
        device = get_first_device_in_args(args)
        if config is None:
            key = _KernelKey(device_ordinal=device.index, params=params)
            best_kernel = self._cache.get(key)
            if best_kernel is None:
                best_kernel = perf_model_cache.predict_best_kernel(kernel_cls, params, device, **kernel_kwargs)
                if best_kernel is None:
                    best_result, best_kernel, results = benchmark_kernel(
                        kernel_cls, params, [args], device=device, **kernel_kwargs
                    )
                    benchmark_logger.log_results(results)
                    benchmark_logger.log_best(best_result)
                self._cache[key] = best_kernel
            return best_kernel
        else:
            return kernel_cls(params, config=config, **kernel_kwargs)
