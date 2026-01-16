"""Implements the CheckerboardSpec class for tensor layout. Use with IndexSpec and TensorSpec."""

from .dim import StaticDim, dim_name_to_dim_or_fold_class_name, BUILTIN_DIM_NAMES
from .dim_arg import DimArg, normalize_dim_arg, extract_dim_from_arg
from .fold import StaticFold
from .gen_specs import GenSpecs
from .derived_dimension import SizedDerivedDimension


class Checkerboard(GenSpecs, SizedDerivedDimension):
    """CUDA / C++ code generator for checkerboard index classes.

    When used with the Generators container, class_name can be omitted and will
    be set from the attribute name.

    Attributes:
        pairs_dim: The dimension for pairs (str, Dim, Fold, StaticDim, or StaticFold).
        colors_dim: The dimension for colors (str, Dim, Fold, StaticDim, or StaticFold).
        class_name: The name of the generated class (optional with Generators).
        offset_dim: The dimension for offset (str, Dim, Fold; default: "LANE").
        size: The size of the checkerboard. If both pairs_dim and colors_dim are
            StaticDim or StaticFold, this is computed automatically as their product.
        ranks: Number of ranks (default: 8).
    """

    def __init__(
        self,
        pairs_dim: DimArg,
        colors_dim: DimArg,
        class_name: str = None,
        offset_dim: DimArg = "LANE",
        size: int = None,
        ranks: int = 8,
    ):
        """Initialize the Checkerboard.

        Args:
            pairs_dim: The dimension for pairs. Can be str, Dim, Fold, StaticDim,
                or StaticFold. If StaticDim/StaticFold, the underlying Dim/Fold
                is extracted and used.
            colors_dim: The dimension for colors. Same types as pairs_dim.
            class_name: The name of the generated class (optional with Generators).
            offset_dim: The dimension for offset (default: "LANE").
            size: The size of the checkerboard. If None and both pairs_dim and
                colors_dim are StaticDim/StaticFold, computed as their product.
            ranks: Number of ranks (default: 8).

        Raises:
            ValueError: If size is None and cannot be computed from pairs/colors,
                or if specified size doesn't match computed size.
        """
        # Extract sizes from StaticDim/StaticFold if available
        pairs_size = None
        colors_size = None

        if isinstance(pairs_dim, (StaticDim, StaticFold)):
            pairs_size = pairs_dim.size
            pairs_dim = extract_dim_from_arg(pairs_dim)

        if isinstance(colors_dim, (StaticDim, StaticFold)):
            colors_size = colors_dim.size
            colors_dim = extract_dim_from_arg(colors_dim)

        # Auto-compute size if both dimensions have sizes
        computed_size = None
        if pairs_size is not None and colors_size is not None:
            computed_size = pairs_size * colors_size

        # Validate or set size
        if size is None:
            if computed_size is not None:
                size = computed_size
            else:
                # Default to 32 for backward compatibility
                size = 32
        elif computed_size is not None and size != computed_size:
            raise ValueError(
                f"Specified size ({size}) doesn't match computed size "
                f"({computed_size} = {pairs_size} * {colors_size}) from "
                f"pairs_dim and colors_dim sizes."
            )

        self.pairs_dim = pairs_dim
        self.colors_dim = colors_dim
        self.class_name = class_name
        self.offset_dim = offset_dim
        self._size = size
        self.ranks = ranks

    @property
    def size(self) -> int:
        """Return the size of the checkerboard."""
        return self._size

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
        pars = ", ".join(
            [
                str(self.ranks),
                pairs_dim_class_name,
                colors_dim_class_name,
                offset_dim_class_name,
            ]
        )
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
