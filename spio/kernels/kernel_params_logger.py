"""Logging functionality for kernel parameters."""
import threading
from contextlib import ContextDecorator
from typing import List, TYPE_CHECKING

from .kernel_key import KernelParams

if TYPE_CHECKING:
    from .kernel_cache import KernelCache

# Global logger for logging kernel parameters
# pylint: disable=C0103
_global_logger = None

# Make it thread-safe.
_global_lock = threading.Lock()


class KernelParamsLogger(ContextDecorator):
    """Context manager for logging kernel parameters."""
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

    def log_params(self, kernel_cache, kernel_factory, params, device, **kernel_kwargs):
        """Log kernel parameters."""
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

    def get_logged_params(self) -> List[KernelParams]:
        """Return the logged kernel parameters."""
        return self.logged_params


def get_global_logger():
    """Get the global logger instance."""
    with _global_lock:
        return _global_logger


def kernel_params_logging_is_enabled():
    """Check if kernel parameters logging is enabled."""
    return get_global_logger() is not None


def log_kernel_params(func):
    """Decorator for conditionally logging function parameters"""

    def wrapper(*args, **kwargs):
        logger = get_global_logger()
        if logger:
            kernel_cache = args[0]
            kernel_factory = args[1]
            params = args[2]
            device = args[3]
            kernel_kwargs = kwargs.copy()
            logger.log_params(kernel_cache, kernel_factory, params, device, **kernel_kwargs)
        return func(*args, **kwargs)

    return wrapper
