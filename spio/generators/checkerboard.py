"""Implements the CheckerboardSpec class for tensor layout. Use with IndexSpec and TensorSpec."""

from dataclasses import dataclass

from .dim import dim_name_to_dim_or_fold_class_name, BUILTIN_DIM_NAMES
from .gen_specs import GenSpecs
from .derived_dimension import SizedDerivedDimension


@dataclass
class Checkerboard(GenSpecs, SizedDerivedDimension):
    """CUDA / C++ code generator for checkerboard index classes.

    When used with the Generators container, class_name can be omitted and will
    be set from the attribute name.

    Attributes:
        pairs_dim: The dimension name for pairs.
        colors_dim: The dimension name for colors.
        class_name: The name of the generated class (optional with Generators).
        offset_dim: The dimension name for offset (default: "LANE").
        ranks: Number of ranks (default: 8).
    """

    pairs_dim: str
    colors_dim: str
    class_name: str = None
    offset_dim: str = "LANE"
    size: int = 32
    ranks: int = 8

    def __post_init__(self):
        """Normalize the dimension names to upper-case."""
        object.__setattr__(self, "pairs_dim", self.pairs_dim.upper())
        object.__setattr__(self, "colors_dim", self.colors_dim.upper())
        object.__setattr__(self, "offset_dim", self.offset_dim.upper())

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
        pairs_dim_class_name = dim_name_to_dim_or_fold_class_name(self.pairs_dim)
        colors_dim_class_name = dim_name_to_dim_or_fold_class_name(self.colors_dim)
        offset_dim_class_name = dim_name_to_dim_or_fold_class_name(self.offset_dim)
        if offset_dim_class_name in BUILTIN_DIM_NAMES:
            offset_dim_class_name = "spio::" + offset_dim_class_name
        pars = f"{self.ranks}, {pairs_dim_class_name}, {colors_dim_class_name}, {offset_dim_class_name}"
        return f"using {self.class_name} = spio::CheckerboardIndex<{pars}>;"

    @property
    def dim_names(self):
        """Return the names of the dimensions."""
        return (self.pairs_dim, self.colors_dim, self.offset_dim)


def header():
    """Return the header file for the checkerboard index."""
    return '#include "spio/checkerboard_index.h"'
