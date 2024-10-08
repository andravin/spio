from typing import Dict

import torch

from spio.kernels.kernel_key import KernelKey

from .. import primary_context_guard

from ..compiler import compile_kernel_configs

from .kernel_util import get_first_device_in_args
from .performance_model_cache import PerformanceModelCache
from .kernel import Kernel
from .kernel_params_logger import log_kernel_params


perf_model_cache = PerformanceModelCache()


class KernelCache:
    """Cache for compiled kernels.

    This class is used to cache compiled kernels for reuse. It also provides a mechanism to
    select the best kernel configuration for a given set of parameters and device.
    If the best kernel is not already in the cache, it will be compiled and loaded.
    """

    def __init__(self):
        self._cache = {}
        self._cache_overlay = {}

    def update_overlay(self, overlay: Dict[str, Kernel]):
        """
        Add a set of kernels to the overlay cache.

        These kernels will be used instead of the main cache. A user may want to use this to
        select specific kernel configurations for benchmarking.

        If an overlay is set, the main cache will not be used.
        """
        self._cache_overlay.update(overlay)

    def clear_overlay(self):
        """Clear the ovleray cache."""
        self._cache_overlay.clear()

    @log_kernel_params
    def get(self, kernel_cls, params, device, **kernel_kwargs) -> Kernel:
        """Return the best kernel for the given params and device.

        If the kernel is not in the cache, it will be compiled and loaded.
        The best kernel configuration is determined by the performance model
        for the device and kernel class.
        """
        key = KernelKey(device_ordinal=device.index, params=params)
        best_kernel = self._cache_overlay.get(key)
        if best_kernel is None:
            if self._cache_overlay:
                raise ValueError(
                    f"Kernel {kernel_cls} with params {params} and device {device} enot found in overlay cache"
                )
            best_kernel = self._cache.get(key)
            if best_kernel is None:
                best_config = perf_model_cache.predict_best_kernel(
                    kernel_cls, params, device, **kernel_kwargs
                )
                if best_config is None:
                    raise ValueError(
                        f"Could not find best config for {kernel_cls} with params {params} and kwargs {kernel_kwargs}"
                    )
                best_kernel = _compile_and_load_kernel(
                    kernel_cls, params, best_config, device, **kernel_kwargs
                )

                self._cache[key] = best_kernel
        return best_kernel


def _compile_and_load_kernel(
    kernel_cls, params, config, device, **kernel_kwargs
) -> Kernel:
    with torch.device(device) as device_obj:
        device_ordinal = device_obj.index if device_obj.index is not None else 0
        arch = torch.cuda.get_device_capability(device=device_obj)
        primary_context_guard.set_device(device_ordinal)
        configs = [config]
        kernels = compile_kernel_configs(
            kernel_cls, params, configs=configs, arch=arch, **kernel_kwargs
        )
        best_kernel = kernels[0]
        device_ordinal = device_obj.index if device_obj.index is not None else 0
        best_kernel.load(device_ordinal=device_ordinal)
        return best_kernel
