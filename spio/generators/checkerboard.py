"""Implements the CheckerboardSpec class for tensor layout. Use with IndexSpec and TensorSpec."""

from dataclasses import dataclass
from typing import Union

from .dim import dim_name_to_dim_or_fold_class_name, BUILTIN_DIM_NAMES
from .dim_arg import DimArg, normalize_dim_arg
from .gen_specs import GenSpecs
from .derived_dimension import SizedDerivedDimension


@dataclass
class Checkerboard(GenSpecs, SizedDerivedDimension):
    """CUDA / C++ code generator for checkerboard index classes.

    When used with the Generators container, class_name can be omitted and will
    be set from the attribute name.

    Attributes:
        pairs_dim: The dimension for pairs (str, Dim, or Fold).
        colors_dim: The dimension for colors (str, Dim, or Fold).
        class_name: The name of the generated class (optional with Generators).
        offset_dim: The dimension for offset (str, Dim, or Fold; default: "LANE").
        ranks: Number of ranks (default: 8).
    """

    pairs_dim: DimArg
    colors_dim: DimArg
    class_name: str = None
    offset_dim: DimArg = "LANE"
    size: int = 32
    ranks: int = 8

    # No __post_init__ - dimension names are resolved lazily in generate() and dim_names

    def _set_class_name(self, name: str) -> None:
        """Set the class name for this checkerboard.

        Called by the Generators container when assigned to an attribute.
        """
        self.class_name = name

    def get_class_name(self) -> str:
        """Return the class name, or None if not set."""
        return self.class_name

    def set_output_dim_name(self, name: str) -> None:
        """Set the output dimension name for this checkerboard."""
        self.offset_dim = name

    def generate(self) -> str:
        """Return the CUDA / C++ source code for the checkerboard index subclass."""
        pairs = normalize_dim_arg(self.pairs_dim)
        colors = normalize_dim_arg(self.colors_dim)
        offset = normalize_dim_arg(self.offset_dim)
        pairs_dim_class_name = dim_name_to_dim_or_fold_class_name(pairs)
        colors_dim_class_name = dim_name_to_dim_or_fold_class_name(colors)
        offset_dim_class_name = dim_name_to_dim_or_fold_class_name(offset)
        if offset_dim_class_name in BUILTIN_DIM_NAMES:
            offset_dim_class_name = "spio::" + offset_dim_class_name
        pars = f"{self.ranks}, {pairs_dim_class_name}, {colors_dim_class_name}, {offset_dim_class_name}"
        return f"using {self.class_name} = spio::CheckerboardIndex<{pars}>;"

    @property
    def dim_names(self):
        """Return the names of the dimensions."""
        return (
            normalize_dim_arg(self.pairs_dim),
            normalize_dim_arg(self.colors_dim),
            normalize_dim_arg(self.offset_dim),
        )


def header():
    """Return the header file for the checkerboard index."""
    return '#include "spio/checkerboard_index.h"'
