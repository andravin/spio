"""Support for CUDA in SPIO.

This module provides functions and classes to compile and manage CUDA kernels.

cuda.nvrtc: Functions to compile CUDA source code.
    This subpackage uses ctypes to interface with the NVRTC library.
cuda.driver: Functions to load kernels from cubins and launch them.
    This subpackage is a Cython extension that encapsulates the CUDA driver API.
"""
