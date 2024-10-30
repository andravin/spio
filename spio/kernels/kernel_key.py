"""Kernel key and parameters for kernel caching."""

from dataclasses import dataclass
from typing import Any, Tuple, TYPE_CHECKING

from .params import Params

if TYPE_CHECKING:
    from .kernel_cache import KernelCache
    from .kernel_factory import KernelFactory


@dataclass(frozen=True)
class KernelKey:
    """A key for a kernel in a KernelCache."""

    device_ordinal: int
    params: Params


@dataclass(frozen=True)
class KernelParams:
    """Kernel parameters captured by KernelParamsLogger."""

    kernel_cache: "KernelCache"
    kernel_factory: "KernelFactory"
    params: Params
    device: Any
    kernel_kwargs: Tuple[str, Any]

    @property
    def key(self) -> KernelKey:
        """Return the kernel key for the params and device."""
        return KernelKey(device_ordinal=self.device.index, params=self.params)
