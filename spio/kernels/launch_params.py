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
        self.grid = _make_3_tuple(self.grid)
        self.block = _make_3_tuple(self.block)


def _make_3_tuple(value: Union[int, Tuple[int, ...]]) -> Tuple[int, int, int]:
    """Convert an int or tuple to a 3-tuple."""
    if isinstance(value, int):
        assert value > 0, "Value must be positive."
        return (value, 1, 1)
    for v in value:
        assert v > 0, "All elements in the tuple must be positive."
    if len(value) == 1:
        return (value[0], 1, 1)
    if len(value) == 2:
        return (value[0], value[1], 1)
    if len(value) == 3:
        return value
    raise ValueError("Value must be an int or a tuple of at most 3 integers.")
