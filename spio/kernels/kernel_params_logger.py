import threading
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Type, Any, List, Tuple, TYPE_CHECKING

from .kernel_key import KernelParams

if TYPE_CHECKING:
    from .kernel_cache import KernelCache

# Global logger for logging kernel parameters
_global_logger = None

# Make it thread-safe.
_global_lock = threading.Lock()


class KernelParamsLogger(ContextDecorator):
    def __enter__(self):
        global _global_logger
        _global_logger = self
        self.logged_params = []
        self.lock = threading.Lock()  # Lock for thread-safe logging
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _global_logger
        _global_logger = None
        return False

    def log_params(self, kernel_cache, kernel_factory, params, device, **kernel_kwargs):
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
        return self.logged_params


def get_global_logger():
    with _global_lock:
        return _global_logger


def kernel_params_logging_is_enabled():
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
