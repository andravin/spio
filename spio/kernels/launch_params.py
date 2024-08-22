from dataclasses import dataclass
from typing import Tuple, Union

import cupy as cp

@dataclass
class LaunchParams:
    grid: Union[int, Tuple[int, ...]]
    block: Union[int, Tuple[int, ...]]

    def __post_init__(self):
        assert self.grid > 0
        assert self.block > 0
        if isinstance(self.grid, int):
            self.grid = (self.grid,)
        if isinstance(self.block, int):
            self.block = (self.block,)
