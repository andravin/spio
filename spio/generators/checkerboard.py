"""Implements the CheckerboardSpec class for tensor layout. Use with IndexSpec and TensorSpec."""

from dataclasses import dataclass

from .dim import dim_name_to_dim_or_fold_class_name


@dataclass
class Checkerboard:
    """CUDA / C++ code generator for checkerboard index classes."""

    class_name: str
    pairs_dim: str
    colors_dim: str
    offset_dim: str
    ranks: int = 8

    def generate(self) -> str:
        """Return the CUDA / C++ source code for the checkerboard index subclass."""
        pairs_dim_class_name = dim_name_to_dim_or_fold_class_name(self.pairs_dim)
        colors_dim_class_name = dim_name_to_dim_or_fold_class_name(self.colors_dim)
        offset_dim_class_name = dim_name_to_dim_or_fold_class_name(self.offset_dim)
        pars = f"{self.ranks}, {pairs_dim_class_name}, {colors_dim_class_name}, {offset_dim_class_name}"
        return f"using {self.class_name} = spio::CheckerboardIndex<{pars}>;"

    @property
    def dim_names(self):
        """Return the names of the dimensions."""
        return (self.pairs_dim, self.colors_dim, self.offset_dim)


def header():
    """Return the header file for the checkerboard index."""
    return '#include "spio/checkerboard_index.h"'
