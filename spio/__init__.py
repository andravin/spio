"""spio: A Python package with efficient GPU kernels for training convolutional neural networks."""

__version__ = "0.4.2"
# Import the CUDA and driver modules to ensure they are initialized
# before accessing their contents.
from .cuda.driver import init, PrimaryContextGuard

# Initialize CUDA driver API
init()

# Retain the primary CUDA context.
primary_context_guard = PrimaryContextGuard()

# Supported GPU architectures
# sm_80: A100
# sm_86: RTX 30 series
# sm_89: RTX 40 series
supported_arch = ["sm_80", "sm_86", "sm_89"]

# sm_90: H100 (not yet supported)
