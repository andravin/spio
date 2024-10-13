from typing import Type, Callable, Union, List

from .kernel import Kernel, get_full_kernel_name
from .launch_params import LaunchParams
from .kernel_cache import KernelCache


class _KernelFactory:
    def __init__(
        self,
        params_cls: Type = None,
        config_cls: Type = None,
        stats_cls: Type = None,
        kernel_name: Union[str, Callable[..., str]] = None,
        configs: Union[List, Callable[..., List]] = None,
        specs: Union[List, Callable[..., List]] = None,
        kernel_source_file: Union[str, Callable[..., str]] = None,
        launch_params: LaunchParams = None,
        src_module: str = "spio.src",
        includes_module: str = "spio.include",
    ):
        self.params_cls = params_cls
        self.config_cls = config_cls
        self.stats_cls = stats_cls
        self._kernel_name = kernel_name
        self._configs = configs
        self._specs = specs
        self._kernel_source_file = kernel_source_file
        self._launch_params = launch_params
        self._kernel_caches = dict()
        self._src_module = src_module
        self._includes_module = includes_module

    def _configs(self, params):
        """Return a generator for the configurations of the given layer parameters."""
        if callable(self._configs):
            return self._configs(params)
        else:
            return self._configs

    def _get_kernel_name(self, **kwargs):
        if callable(self._kernel_name):
            return self._kernel_name(**kwargs)
        else:
            return self._kernel_name

    def _get_full_kernel_name(self, params, **kwargs):
        kernel_name = self._get_kernel_name(**kwargs)
        return get_full_kernel_name(kernel_name, params)

    def _get_specs(self, params, config, **kwargs):
        if callable(self._specs):
            return self._specs(params, config, **kwargs)
        else:
            return self._specs, self._launch_params

    def get_kernel_cache(self, **kwargs):
        kernel_name = self._get_kernel_name(**kwargs)
        kernel_cache = self._kernel_caches.get(kernel_name)
        if kernel_cache is None:
            kernel_cache = KernelCache()
            self._kernel_caches[kernel_name] = kernel_cache
        return kernel_cache

    def get_kernel(self, params, device, **kwargs):
        kernel_cache = self.get_kernel_cache(**kwargs)
        return kernel_cache.get(self, params, device, **kwargs)

    def make_kernel(self, params, config, **kwargs):
        kernel_name = self._get_full_kernel_name(params, **kwargs)
        specs, launch_params = self._get_specs(params, config, **kwargs)
        return Kernel(
            kernel_name,
            launch_params,
            kernel_source_file=self._kernel_source_file,
            specs=specs,
            params=params,
            config=config,
            src_module=self._src_module,
            includes_module=self._includes_module,
        )


def make_kernel_factory(
    params_cls: Type = None,
    config_cls: Type = None,
    stats_cls: Type = None,
    kernel_name: Union[str, Callable[..., str]] = None,
    configs: Union[List, Callable[..., List]] = None,
    specs: Union[List, Callable[..., List]] = None,
    kernel_source_file: Union[str, Callable[..., str]] = None,
    launch_params: Union[LaunchParams, Callable[..., LaunchParams]] = None,
    src_module: str = "spio.src",
    includes_module: str = "spio.include",
):
    return _KernelFactory(
        params_cls=params_cls,
        config_cls=config_cls,
        stats_cls=stats_cls,
        kernel_name=kernel_name,
        configs=configs,
        specs=specs,
        kernel_source_file=kernel_source_file,
        launch_params=launch_params,
        src_module=src_module,
        includes_module=includes_module,
    )
