"""This file implements the Dims class."""

from typing import Dict, Generator, List, Tuple, Union

from .derived_dimension import DerivedDimension, SizedDerivedDimension

# Type alias for dimension values: either an integer size or a derived dimension generator
DimValue = Union[int, DerivedDimension]


def _parse_dim_name_and_fold(name: str) -> Tuple[str, int]:
    """Parse a dimension name into base name and fold factor.

    Args:
        name: Dimension name like 'K8', 'K4', 'K', 'I32', etc.

    Returns:
        Tuple of (base_name, fold_factor). fold_factor is 1 if not specified.
    """
    fold_factor = 1
    base_name = name
    for i, char in enumerate(name):
        if char.isdigit():
            fold_factor = int(name[i:])
            base_name = name[:i]
            break
    return base_name, fold_factor


def _compute_fold_sizes(
    dims: Dict[str, DimValue],
) -> Dict[str, DimValue]:
    """Compute automatic fold sizes for dimensions specified as -1.

    Args:
        dims: Dictionary of dimension names to sizes. Size of -1 indicates
            automatic computation.

    Returns:
        Dictionary with all -1 sizes replaced by computed values.

    Raises:
        ValueError: If the coarsest fold has size -1, or if an explicit size
            doesn't match the computed value.
    """
    # Group dimensions by base name
    base_dim_folds: Dict[str, List[Tuple[str, int, DimValue]]] = {}
    for name, size in dims.items():
        base_name, fold_factor = _parse_dim_name_and_fold(name)
        if base_name not in base_dim_folds:
            base_dim_folds[base_name] = []
        base_dim_folds[base_name].append((name, fold_factor, size))

    result = dict(dims)

    for base_name, folds in base_dim_folds.items():
        # Sort by fold factor descending (coarsest first)
        folds_sorted = sorted(folds, key=lambda x: x[1], reverse=True)

        # Coarsest fold must have explicit size
        coarsest_name, coarsest_fold, coarsest_size = folds_sorted[0]
        if coarsest_size == -1:
            raise ValueError(
                f"Coarsest fold '{coarsest_name}' (fold factor {coarsest_fold}) "
                f"must have an explicit size, not -1."
            )

        # Compute sizes for remaining folds
        for i in range(1, len(folds_sorted)):
            name, fold_factor, specified_size = folds_sorted[i]
            prev_name, prev_fold, _prev_size = folds_sorted[i - 1]

            # Computed size = ratio of fold factors
            computed_size = prev_fold // fold_factor

            if prev_fold % fold_factor != 0:
                raise ValueError(
                    f"Fold factor {prev_fold} of '{prev_name}' is not divisible "
                    f"by fold factor {fold_factor} of '{name}'."
                )

            if specified_size == -1:
                result[name] = computed_size
            elif specified_size != computed_size:
                raise ValueError(
                    f"Dimension '{name}' has explicit size {specified_size}, "
                    f"but computed size is {computed_size} "
                    f"(from {prev_fold}/{fold_factor})."
                )

    return result


class Dims:
    """A class to represent the dimensions of a tensor."""

    def __init__(self, **dims: Dict[str, DimValue]):
        """Initialize the Dims object with the given dimensions and sizes.

        Dimensions with size -1 will have their sizes computed automatically based
        on fold factor ratios. The coarsest fold must have an explicit size.

        Example:
            # Create a Dims object with folds and automatic size computation.
            # k8=16 is the coarsest fold with explicit size 16.
            # k4=-1 computes to size 2 (8/4 = 2).
            # k=-1 computes to size 4 (4/1 = 4).
            # Total K dimension: 16 * 2 * 4 = 128
            dims = Dims(k8=16, i=32, k4=-1, k=-1)

        Args:
            **dims: Keyword arguments representing the dimensions. Each dimension
                is specified as a name-value pair, where the name is a string
                and the value is an integer or a DerivedDimension.

                Names can include fold factors (e.g., 'k8', 'k4', 'k'). K8 means a fold
                of dimension K with fold factor 8.

                Names are automatically normalized to upper-case.

                The coarsest folds must have an explicit size, the rest can use
                a value of -1 to indicate the size should be computed automatically
                from fold factor ratios.
        """
        normalized = {key.upper(): value for key, value in dims.items()}
        self._dims = _compute_fold_sizes(normalized)

    def items(self) -> Generator[Tuple[str, DimValue], None, None]:
        """Get the dimensions as a generator of (name, value) pairs."""
        return self._dims.items()

    def keys(self) -> Generator[str, None, None]:
        """Get the names of the dimensions."""
        return self._dims.keys()

    def values(self) -> Generator[DimValue, None, None]:
        """Get the values of the dimensions."""
        return self._dims.values()

    def __getitem__(self, key) -> DimValue:
        """Get the value of a dimension by its name."""
        return self._dims[key]

    def __contains__(self, key) -> bool:
        """Check if a dimension is present in the Dims object."""
        return key in self._dims


class Strides:
    """A class to represent the strides of a tensor."""

    def __init__(self, **strides: Dict[str, int]):
        """Initialize the Strides object with the given strides.

        Args:
            **strides: Keyword arguments representing the strides. Each stride
                is specified as a name-value pair, where the name is a string
                and the value is an integer.
        """
        self._strides = {key.upper(): value for key, value in strides.items()}

    def items(self) -> Generator[Tuple[str, int], None, None]:
        """Get the strides as a generator of (name, value) pairs."""
        return self._strides.items()

    def keys(self) -> Generator[str, None, None]:
        """Get the names of the strides."""
        return self._strides.keys()

    def values(self) -> Generator[int, None, None]:
        """Get the values of the strides."""
        return self._strides.values()

    def __getitem__(self, key) -> int:
        """Get the value of a stride by its name."""
        return self._strides[key]

    def __contains__(self, key) -> bool:
        """Check if a stride is present in the Strides object."""
        return key in self._strides


def compute_full_strides(
    dims: Dims,
    given_strides: Strides,
) -> Dict[str, int]:
    """Compute the full strides for the given dimensions."""
    if given_strides is None:
        given_strides = {}
    all_strides = {}
    stride = 1
    for name, value in reversed(dims.items()):
        if name in given_strides:
            stride = given_strides[name]
        all_strides[name] = stride
        # Compute the default stride of the next dimension.
        if isinstance(value, SizedDerivedDimension):
            dim_size = value.size
        else:
            dim_size = value
        stride *= dim_size
    return all_strides
