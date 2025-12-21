"""
Enumeration of built-in variables for GPU programming.
"""

from enum import Enum


class BuiltIn(Enum):
    """Enumeration of built-in variables for GPU programming.

    These values can be used as initializaters for Fold or CompoundIndex generators.
    """

    BLOCK_IDX_X = "blockIdx.x"
    BLOCK_IDX_Y = "blockIdx.y"
    BLOCK_IDX_Z = "blockIdx.z"
    THREAD_IDX_X = "threadIdx.x"
    THREAD_IDX_Y = "threadIdx.y"
    THREAD_IDX_Z = "threadIdx.z"
