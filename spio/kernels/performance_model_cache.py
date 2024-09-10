from pathlib import Path
from dataclasses import dataclass

import xgboost as xgb
import torch
import appdirs

from ..util import (
    get_cache_dir,
    params_and_configs_to_dataframe,
    get_formatted_device_name,
    get_formatted_arch,
)
from ..cuda import primary_context_guard
from ..compiler import compile_kernel_configs

PERFORMANCE_MODEL_EXTENSION = ".ubj"


@dataclass(frozen=True)
class _PerformanceModelKey:
    kernel_name: str
    device_name: str


class PerformanceModelCache:
    def __init__(self):
        self._cache = {}
        self._no_cache = {}

    def predict_best_kernel(self, kernel_cls, params, device, **kernel_kwargs):
        kernel_name = kernel_cls.get_kernel_name(**kernel_kwargs)
        device_name = get_formatted_device_name(device)
        arch = get_formatted_arch(device)
        performance_model = self._get_performance_model(kernel_name, device_name, arch)
        if performance_model is None:
            return None

        configs = list(kernel_cls.configs(params))
        best_config = self._predict_best_config(performance_model, params, configs)
        if best_config is None:
            return None

        with torch.device(device) as device_obj:
            device_ordinal = device_obj.index if device_obj.index is not None else 0
            arch = torch.cuda.get_device_capability(device=device_obj)
            primary_context_guard.set_device(device_ordinal)
            configs = [best_config]
            kernels = compile_kernel_configs(
                kernel_cls, params, configs=configs, arch=arch, **kernel_kwargs
            )
            best_kernel = kernels[0]
            device_ordinal = device_obj.index if device_obj.index is not None else 0
            best_kernel.load(device_ordinal=device_ordinal)
            return best_kernel

    def _predict_best_config(self, performance_model, params, configs):
        df = params_and_configs_to_dataframe(params, configs)
        dm = xgb.DMatrix(df)
        predictions = performance_model.predict(dm)
        best_config = configs[predictions.argmin()]
        return best_config

    def _get_performance_model(self, kernel_name, device, arch):
        key = _PerformanceModelKey(kernel_name, device)
        performance_model = self._cache.get(key)
        if performance_model is None:
            if self._no_cache.get(key):
                performance_model = None
            else:
                performance_model = self._load_performance_model(
                    kernel_name, device, arch
                )
                if performance_model is not None:
                    self._cache[key] = performance_model
                else:
                    self._no_cache[key] = True
        return performance_model

    def _load_performance_model(self, kernel_name, device, arch):
        cache_dir = get_cache_dir()
        filename = get_performance_model_file_name(kernel_name, device, arch)
        path = Path(cache_dir, filename)
        if not path.exists():
            return None
        else:
            model = xgb.Booster()
            model.load_model(str(path))
            return model


def get_performance_model_file_name(
    kernel: str, device: str, arch: str, ext: str = PERFORMANCE_MODEL_EXTENSION
):
    """Return the performance model filename for the given kernel, device, and architecture."""
    return f"perfmodel__{kernel}__{device}__{arch}{ext}"
