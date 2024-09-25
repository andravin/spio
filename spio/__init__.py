__version__ = "0.1.0"

# Import the CUDA and driver modules to ensure they are initialized
# before accessing their contents.
from .cuda.driver import init, PrimaryContextGuard

# Initialize CUDA driver API
init()

# Retain the primary CUDA context.
primary_context_guard = PrimaryContextGuard()
