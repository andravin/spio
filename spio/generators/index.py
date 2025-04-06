"""Code generator for custom index classes in CUDA / C++."""

from typing import List, Generator
from dataclasses import dataclass

from .dims import Dims, Strides, compute_full_strides
from .dim import dim_name_to_dim_or_fold_class_name


@dataclass
class Index:
    """CUDA Code generator for custom index classes.

    This class is used to generate custom index classes that map linear offsets
    to multidimensional coordinates.

    Optionally, specify the strides for any dimensions that are not contiguous in memory.

    Also, specify in the dummies list the names of any dimensions that should not be applied
    to tensor subscript operators. For example, a "repeat" dimension that exists only to cause
    the inner dimension to repeat values across the lanes of a warp can be specified as a dummy,
    and then the "repeat" dimension will not be used when applying the index to the tensor subscript
    operator like "tensor[index]".

    Attributes:
        class_name (str): The name of the custom index class.
        dims (Dims): A dictionary mapping dimension names to their sizes.
        strides (Strides): Optional strides for the dimensions.
        dummies (List[str]): List of dimension names that will not be applied to tensor subscripts.
    """

    class_name: str
    dims: Dims
    strides: Strides = None
    dummies: List[str] = None

    def __post_init__(self):
        # Ensure strides are calculated for each dimension
        self.strides = compute_full_strides(self.dims, self.strides)
        self.dummies = self.dummies or []

    def generate_with_context(self, user_data_types: List[str] = None) -> str:
        """Generate the C++ source code for the custom index class."""
        return _generate_index(self.class_name, self.dims, self.strides, self.dummies)

    @property
    def total_size(self) -> int:
        """Total number of elements (product of all dimension sizes)."""
        product = 1
        for size in self.dims.values():
            product *= size
        return product

    @property
    def dim_names(self) -> Generator[str, None, None]:
        """Return the names of the dimensions in the index."""
        for name, _ in self.dims.items():
            yield name


def header() -> str:
    """Return a C++ statement that includes the spio index header.

    The header implements the C++ base template classes from which the
    custom index classes inherit.
    """
    return '#include "spio/index.h"'


def _generate_index(
    class_name: str,
    dims: Dims,
    strides: Strides,
    dummy_dims: List[str] = None,
) -> str:
    """Generate a using statement for an Index template instantiation."""
    dim_infos = []
    specializations = []

    # Generate DimInfo parameters for each dimension
    for name, size_value in dims.items():
        # Handle the size (now all integers)
        size_str = str(size_value)

        # Get the stride for this dimension
        stride = strides[name]

        # Use dim_name_to_dim_or_fold_class_name to handle both regular and fold dimensions
        dim_class = dim_name_to_dim_or_fold_class_name(name)

        # Add the DimInfo parameter
        dim_infos.append(f"spio::DimInfo<{dim_class}, {size_str}, {stride}>")
        
        # If this is a dummy dimension, generate a specialization
        if name in (dummy_dims or []):
            dim_info = f'spio::DimInfo<{dim_class}, {size_str}, {stride}>'
            specializations.append(
                f"namespace spio {{ namespace detail {{\n"
                f"    template<> struct is_dummy_dimension<{dim_info}> {{\n"
                f"        static constexpr bool value = true;\n"
                f"    }};\n"
                f"}}}}\n"
            )

    # Generate the index type using statement
    index_using = f"using {class_name} = spio::Index<{', '.join(dim_infos)}>;\n"

    # Combine the using statement with any specializations
    return index_using + "\n".join(specializations)
