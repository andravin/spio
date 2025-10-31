"""Kernel cache for compiled kernels."""

from typing import Dict

import torch

from .. import primary_context_guard

from ..compiler import compile_kernel_configs
from ..util import logger_enabled, get_device_ordinal
from ..cuda.driver import get_device_attributes

from .kernel_key import KernelKey
from .performance_model_cache import PerformanceModelCache
from .kernel import Kernel
from .kernel_params_logger import _log_kernel_params, kernel_params_logging_is_enabled

perf_model_cache = PerformanceModelCache()


class KernelCache:
    """Cache for compiled kernels.

    This class is used to cache compiled kernels for reuse. It also
    provides a mechanism to select the best kernel configuration for a
    given set of parameters and device. If the best kernel is not
    already in the cache, it will be compiled and loaded.
    """

    def __init__(self):
        self._cache = {}
        self._cache_overlay = {}

    def update_overlay(self, overlay: Dict[str, Kernel]):
        """Add a set of kernels to the overlay cache.

        These kernels will be used instead of the main cache. A user may
        want to use this to select specific kernel configurations for
        benchmarking.

        If an overlay is set, the main cache will not be used.
        """
        self._cache_overlay.update(overlay)

    def clear_overlay(self):
        """Clear the ovleray cache."""
        self._cache_overlay.clear()

    def clear_cache(self):
        """Clear the main cache."""
        self._cache.clear()

    @_log_kernel_params
    def get(self, kernel_factory, params, device, **kernel_kwargs) -> Kernel:
        """Return the best kernel for the given params and device.

        If the kernel is not in the cache, it will be compiled and
        loaded. The best kernel configuration is determined by the
        performance model for the device and kernel class.
        """
        key = KernelKey(device_ordinal=device.index, params=params)
        best_kernel = self._cache_overlay.get(key)
        if best_kernel is None:
            if self._cache_overlay:
                raise ValueError(
                    f"Kernel {kernel_factory} with params {params} and device {device} "
                    f"not found in overlay cache"
                )
            best_kernel = self._cache.get(key)
            if best_kernel is None:
                if kernel_params_logging_is_enabled():
                    # If kernel params are being logged, the performance model
                    # might not exist for this kernel.
                    # We also don't care about performance in this case.
                    # So we just use the first configuration.
                    # Be careful to clear the cache when you are done logging params so
                    # that the first config is not used for performance.
                    device_idx = get_device_ordinal(device)
                    device_attr = get_device_attributes(device_idx)
                    configs = kernel_factory.configs(params, device_attr, **kernel_kwargs)
                    best_config = next(configs)
                else:
                    best_config = perf_model_cache.predict_best_kernel(
                        kernel_factory, params, device, **kernel_kwargs
                    )
                best_kernel = _compile_and_load_kernel(
                    kernel_factory, params, best_config, device, **kernel_kwargs
                )
                self._cache[key] = best_kernel
                if logger_enabled:
                    print(
                        f"spio: compiled kernel {best_kernel.kernel_name} for device {device}."
                    )
        return best_kernel


def _compile_and_load_kernel(
    kernel_factory, params, config, device, **kernel_kwargs
) -> Kernel:
    with torch.device(device) as device_obj:
        device_ordinal = device_obj.index if device_obj.index is not None else 0
        device_attr = get_device_attributes(device_ordinal)
        primary_context_guard.set_device(device_ordinal)
        configs = [config]
        kernels = compile_kernel_configs(
            kernel_factory,
            params,
            configs=configs,
            device_attr=device_attr,
            **kernel_kwargs,
        )
        best_kernel = kernels[0]
        device_ordinal = device_obj.index if device_obj.index is not None else 0
        best_kernel.load(device_ordinal=device_ordinal)
        return best_kernel
