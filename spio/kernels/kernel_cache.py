from dataclasses import dataclass

from .benchmark import benchmark
from .kernel_util import get_first_device_in_args

@dataclass(frozen=True)
class _KernelKey:
    device: str
    params: object


class KernelCache:
    def __init__(self):
        self._cache = {}

    def get(self, kernel_cls, params, args, config=None, **kwargs):
        device = get_first_device_in_args(args)
        if config is None:
            key = _KernelKey(device=device, params=params)
            kernel = self._cache.get(key)
            if kernel is None:
                best = benchmark(kernel_cls, params, [args], device=device, **kwargs)
                kernel = best.kernel
                self._cache[key] = kernel
            return kernel
        else:
            return kernel_cls(params, config=config, **kwargs)

