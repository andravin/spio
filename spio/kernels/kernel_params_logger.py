"""Logging functionality for kernel parameters."""

import threading
from contextlib import ContextDecorator
from typing import List, TYPE_CHECKING, Any

from .kernel_key import KernelParams
from .params import Params

if TYPE_CHECKING:
    from .kernel_cache import KernelCache
    from .kernel_factory import KernelFactory

# Global logger for logging kernel parameters
# pylint: disable=C0103
_global_logger = None

# Make it thread-safe.
_global_lock = threading.Lock()


class KernelParamsLogger(ContextDecorator):
    """Context manager for logging kernel parameters.

    This class logs the Params passed to kernel functions. It is useful
    for gathering the Spio layers used in a model.

    Example:
        with KernelParamsLogger() as logger:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(inputs[0])
            out.sum().backward()
            logged_kernel_params = logger.get_logged_params()
    """

    def __init__(self):
        self.logged_params = []
        self.lock = None

    def __enter__(self):
        # pylint: disable=W0603
        global _global_logger
        _global_logger = self
        self.logged_params = []
        self.lock = threading.Lock()  # Lock for thread-safe logging
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # pylint: disable=W0603
        global _global_logger
        _global_logger = None
        return False

    def get_logged_params(self) -> List[KernelParams]:
        """Return the logged kernel parameters."""
        return self.logged_params

    def _log_params(
        self,
        kernel_cache: "KernelCache",
        kernel_factory: "KernelFactory",
        params: Params,
        device: Any,
        **kernel_kwargs
    ):
        """Log kernel parameters.

        Used internally by the _log_kernel_params() decorator.
        """
        with self.lock:
            self.logged_params.append(
                KernelParams(
                    kernel_cache,
                    kernel_factory,
                    params,
                    device,
                    tuple(kernel_kwargs.items()),
                )
            )


def get_global_logger() -> KernelParamsLogger:
    """Get the global logger instance."""
    with _global_lock:
        return _global_logger


def kernel_params_logging_is_enabled() -> bool:
    """Check if kernel parameters logging is enabled."""
    return get_global_logger() is not None


def _log_kernel_params(func):
    """Decorator for conditionally logging function parameters.

    Used internally by spio for logging the kernel parameters passed
    to kernel functions.
    """

    def wrapper(*args, **kwargs):
        logger = get_global_logger()
        if logger:
            kernel_cache = args[0]
            kernel_factory = args[1]
            params = args[2]
            device = args[3]
            kernel_kwargs = kwargs.copy()
            logger._log_params(
                kernel_cache, kernel_factory, params, device, **kernel_kwargs
            )
        return func(*args, **kwargs)

    return wrapper
