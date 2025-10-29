"""Kernel factory for creating Kernel objects for CUDA kernels."""

from typing import Type, Callable, Union, List, Any, TypeVar

import torch

from ..util import check_channels_last
from ..cuda.driver import DeviceAttributes

from .params import Params
from .stats import Stats
from .kernel_cache import KernelCache
from .kernel import Kernel, KernelSpec, get_full_kernel_name

# A kernel tile configuration class. Each kernel defines its own configuration dataclass.
Config = TypeVar("Config")


KernelNameCallback = Callable[[Any], str]
"""Callback that returns the kernel name.

Args:
    kwargs: Keyword arguments that can be used to determine the kernel name.

Returns:
    str: The kernel name.
"""

KernelSourceFileCallback = Callable[[Params, Any], str]
"""Callback that returns the kernel source filename.

Args:
    params: The layer parameters.
    kwargs: Keyword arguments that can be used to determine the kernel source filename.

Returns:
    str: The kernel source filename.
"""


ConfigsCallback = Callable[[Params, DeviceAttributes, Any], List[Config]]
"""Callback that returns a list of kernel configurations.

Args:
    params: The layer parameters.
    kwargs: Keyword arguments that can be used to determine the kernel configurations.

Returns:
    List[Config]: A list of kernel configurations.
"""

KernelSpecCallback = Callable[[Params, Config, DeviceAttributes, Any], KernelSpec]
"""Callback that returns the kernel specifications.

Args:
    params: The layer parameters.
    config: The kernel configuration.
    kwargs: Keyword arguments that can be used to determine the kernel specifications.

Returns:
    KernelSpec: The kernel specifications.
"""


ArgsChecker = Callable[[List[torch.Tensor]], None]
"""Callback that checks the arguments for the kernel.

Args:
    args: List[torch.Tensor]: The arguments to check.

Raises:
    AssertionError: If the arguments are not valid.

Returns:
    None
"""


class KernelFactory:
    """Factory for creating Kernel objects for a CUDA kernel.

    Use the make_kernel() method to create a new kernel object for a
    given layer parameters and kernel configuration.

    Use the get_kernel() method to get the best kernel for a given set
    of layer parameters and device. It returns a cached kernel if one is
    found, otherwise it estimates the best kernel configuration using
    the kernel's performance model and compiles a new kernel that uses
    it.

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
        kernel_spec: Union[KernelSpec, KernelSpecCallback] = None,
        kernel_source_file: Union[str, KernelSourceFileCallback] = None,
        src_module: str = "spio.src",
        includes_module: str = "spio.include",
        perf_model_skip_params: List[str] = None,
        args_checker: ArgsChecker = check_channels_last,
    ):
        """Initialize the kernel factory.

        The initializer configures a new kernel factory object to compile instances
        of a kernel.

        Arguments:
            params_cls: The layer parameters class.
            config_cls: The kernel configuration class.
            stats_cls: The kernel statistics class.
            kernel_name: The name of the kernel or a callback that returns it.
            configs: A list of kernel configurations or a callback that returns it.
            kernel_spec: The kernel specifications or a callback that returns it.
            kernel_source_file: The kernel source filename or a callback that returns it.
            src_module: The source module for the kernel.
            includes_module: The includes module for the kernel.
            perf_model_skip_params: List of parameter names to skip in the performance model.
            args_checker: A callback that checks the arguments for the kernel.
        """
        if perf_model_skip_params is None:
            perf_model_skip_params = []
        self.params_cls = params_cls
        self.config_cls = config_cls
        self.stats_cls = stats_cls
        self._kernel_name = kernel_name
        self._configs = configs
        self._kernel_spec = kernel_spec
        self._kernel_source_file = kernel_source_file
        self._kernel_caches = {}
        self._src_module = src_module
        self._includes_module = includes_module
        self.per_model_skip_params = perf_model_skip_params
        self._args_checker = args_checker

    def configs(
        self, params: Params, device_attr: DeviceAttributes, **kwargs
    ) -> List[Config]:
        """Return all configs of the given layer parameters."""
        if callable(self._configs):
            return self._configs(params, device_attr, **kwargs)
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

    def get_kernel_spec(
        self, params: Params, config: Config, device_attr: DeviceAttributes, **kwargs
    ) -> KernelSpec:
        """Return the kernel specs and launch parameters.

        Kernel specs are code generators for named tensors, constant
        variables, macros, and other kernel-specific structures that are
        used in the CUDA kernel source code.

        Args:
            params: The layer parameters.
            config: The kernel configuration.
        """
        if callable(self._kernel_spec):
            return self._kernel_spec(params, config, device_attr, **kwargs)
        return self._kernel_spec

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

    def make_kernel(
        self, params: Params, config, device_attr: DeviceAttributes, **kwargs
    ) -> Kernel:
        """Return a new Kernel object for the params and config."""
        kernel_name = self.get_full_kernel_name(params, **kwargs)
        kernel_specs = self.get_kernel_spec(params, config, device_attr, **kwargs)
        return Kernel(
            kernel_name,
            kernel_specs,
            kernel_source_file=self._kernel_source_file,
            params=params,
            config=config,
            src_module=self._src_module,
            includes_module=self._includes_module,
            args_checker=self._args_checker,
        )
