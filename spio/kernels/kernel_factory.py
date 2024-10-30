"""Kernel factory for creating Kernel objects for CUDA kernels."""

from typing import Type, Callable, Union, List, Any, TypeVar, Tuple

from ..generators.gen_specs import GenSpecs
from .params import Params
from .kernel import Kernel, get_full_kernel_name
from .launch_params import LaunchParams
from .kernel_cache import KernelCache
from .stats import Stats


# A kernel tile configuration class. Each kernel defines its own configuration dataclass.
Config = TypeVar("Config")


# Function prototypes for KernelFactory callbacks.
KernelNameCallback = Callable[[Any], str]
KernelSourceFileCallback = Callable[[Params, Any], str]
ConfigsCallback = Callable[[Params, Any], List[Config]]
SpecsCallback = Callable[[Params, Config, Any], List[GenSpecs]]


def make_kernel_factory(
    params_cls: Type = None,
    config_cls: Type = None,
    stats_cls: Type = None,
    kernel_name: Union[str, KernelNameCallback] = None,
    configs: Union[List[Config], ConfigsCallback] = None,
    specs: Union[List[GenSpecs], SpecsCallback] = None,
    kernel_source_file: Union[str, KernelSourceFileCallback] = None,
    launch_params: LaunchParams = None,
    src_module: str = "spio.src",
    includes_module: str = "spio.include",
    perf_model_skip_params: List[str] = None,
) -> "KernelFactory":
    """Return a new KernelFactory object for a CUDA kernel.

    All callable arguments must be functions that take (params, **kwargs)
    arguments and return the corresponding values.

    Args:
        params_cls: The class for the layer parameters.
        config_cls: The class for the kernel configuration.
        stats_cls: The class for the kernel statistics.
        kernel_name: The name of the kernel
          or a function that returns it.
        configs: The list of kernel configurations
          or a function that returns them.
        specs: The list of kernel specs or a function
          function that returns the specs and launch parameters.
        kernel_source_file: The file name of the CUDA kernel source code.
        launch_params: Optional launch parameters for the kernel.
          These may be returned by the specs function instead.
        src_module: The module name where the kernel_source_file is located.
        includes_module: The module name where any include files are located.
    """
    return KernelFactory(
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
        perf_model_skip_params=perf_model_skip_params,
    )


class KernelFactory:
    """Factory for creating Kernel objects for a CUDA kernel.

    Use the make_kernel() method to create a new kernel object for a
    given layer parameters and kernel configuration.

    Use the get_kernel() method to get the best kernel for a given set
    of layer parameters and device. It returns a cached kernel if one is
    found, otherwise it estimates the best kernel configuration using
    the kernel's performance model and compiles a new kernel that uses
    it.

    Users do not instantiate this class directly. Rather, they define new
    kernels by calling the make_kernel_factory() function.

    Methods that take keyword arguments (**kwargs) use them to
    distinguish between different modes of the kernel. For example, a forward
    and backward kernel may use the keyword argument (igrad:bool=False)
    to differentiate between FPROP and BPROP versions of the kernel.
    """

    def __init__(
        self,
        params_cls: Type[Params] = None,
        config_cls: Type[Config] = None,
        stats_cls: Type[Stats] = None,
        kernel_name: Union[str, KernelNameCallback] = None,
        configs: Union[List[Config], ConfigsCallback] = None,
        specs: Union[List[GenSpecs], SpecsCallback] = None,
        kernel_source_file: Union[str, KernelSourceFileCallback] = None,
        launch_params: LaunchParams = None,
        src_module: str = "spio.src",
        includes_module: str = "spio.include",
        perf_model_skip_params: List[str] = None,
    ):
        if perf_model_skip_params is None:
            perf_model_skip_params = []
        self.params_cls = params_cls
        self.config_cls = config_cls
        self.stats_cls = stats_cls
        self._kernel_name = kernel_name
        self._configs = configs
        self._specs = specs
        self._kernel_source_file = kernel_source_file
        self._launch_params = launch_params
        self._kernel_caches = {}
        self._src_module = src_module
        self._includes_module = includes_module
        self.per_model_skip_params = perf_model_skip_params

    def configs(self, params: Params, **kwargs) -> List[Config]:
        """Return all configs of the given layer parameters."""
        if callable(self._configs):
            return self._configs(params, **kwargs)
        return self._configs

    def get_kernel_name(self, **kwargs) -> str:
        """The name of the kernel with the keyword args."""
        if callable(self._kernel_name):
            return self._kernel_name(**kwargs)
        return self._kernel_name

    def get_full_kernel_name(self, params: Params, **kwargs) -> str:
        """Return the full name of the kernel.

        The full name includes the kernel name and the parameters.
        """
        kernel_name = self.get_kernel_name(**kwargs)
        return get_full_kernel_name(kernel_name, params)

    def get_specs(
        self, params: Params, config: Config, **kwargs
    ) -> Tuple[List[GenSpecs], LaunchParams]:
        """Return the kernel specs and launch parameters.

        Kernel specs are code generators for named tensors, constant
        variables, macros, and other kernel-specific structures that are
        used in the CUDA kernel source code.

        Args:
            params: The layer parameters.
            config: The kernel configuration.
        """
        if callable(self._specs):
            return self._specs(params, config, **kwargs)
        return self._specs, self._launch_params

    def get_kernel_cache(self, **kwargs) -> KernelCache:
        """Return the kernel cache for the given keryword arguments."""
        kernel_name = self.get_kernel_name(**kwargs)
        kernel_cache = self._kernel_caches.get(kernel_name)
        if kernel_cache is None:
            kernel_cache = KernelCache()
            self._kernel_caches[kernel_name] = kernel_cache
        return kernel_cache

    def get_kernel(self, params: Params, device, **kwargs) -> Kernel:
        """Return the best kernel for the layer parameters and device.

        Returns a cached kernel if one is found matching the params and
        device. Otherwise, uses the kernel's performance model to
        estimate the best kernel configuration and compiles a new kernel
        that uses it.
        """
        kernel_cache = self.get_kernel_cache(**kwargs)
        return kernel_cache.get(self, params, device, **kwargs)

    def make_kernel(self, params: Params, config, **kwargs) -> Kernel:
        """Return a new Kernel object for the params and config."""
        kernel_name = self.get_full_kernel_name(params, **kwargs)
        specs, launch_params = self.get_specs(params, config, **kwargs)
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
