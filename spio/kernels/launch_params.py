"""A dataclass for defining launch parameters for CUDA kernels."""

from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class LaunchParams:
    """Launch parameters for CUDA kernels.

    Attributes:
        grid: The grid size.
        block: The block size.
        shared_mem_bytes: The amount of dynamic shared memory in bytes.
    """

    grid: Union[int, Tuple[int, ...]]
    block: Union[int, Tuple[int, ...]]
    shared_mem_bytes: int = 0

    def __post_init__(self):
        assert self.grid > 0
        assert self.block > 0
        if isinstance(self.grid, int):
            self.grid = (self.grid, 1, 1)
        if isinstance(self.block, int):
            self.block = (self.block, 1, 1)
