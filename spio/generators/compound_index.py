"""Code generator for custom index classes in CUDA / C++."""

from typing import List, Generator
from dataclasses import dataclass

from .dims import Dims, Strides, compute_full_strides
from .dim import dim_name_to_dim_or_fold_class_name, BUILTIN_DIM_NAMES
from .built_in import BuiltIn
from .gen_specs import GenSpecsWithContext


@dataclass
class CompoundIndex(GenSpecsWithContext):
    """CUDA Code generator for custom index classes.

    This class is used to generate custom index classes that map linear offsets
    to multidimensional coordinates.

    Optionally, specify the strides for any dimensions that are not contiguous in memory.

    Also, specify in the dummies list the names of any dimensions that should not be applied
    to tensor subscript operators. For example, a "repeat" dimension that exists only to cause
    the inner dimension to repeat values across the lanes of a warp can be specified as a dummy,
    and then the "repeat" dimension will not be used when applying the index to the tensor subscript
    operator like "tensor[index]".

    When used with the Generators container, class_name can be omitted and will
    be set from the attribute name.

    Attributes:
        dims (Dims): A dictionary mapping dimension names to their sizes.
        class_name (str): The name of the custom index class (optional with Generators).
        strides (Strides): Optional strides for the dimensions.
        dummies (List[str]): List of dimension names that will not be applied to tensor subscripts.
    """

    dims: Dims
    class_name: str = None
    strides: Strides = None
    dummies: List[str] = None
    init: BuiltIn = None

    def __post_init__(self):
        # Ensure strides are calculated for each dimension
        if isinstance(self.dims, dict):
            self.dims = Dims(**self.dims)
        if isinstance(self.strides, dict):
            self.strides = Strides(**self.strides)
        self.strides = compute_full_strides(self.dims, self.strides)
        self.dummies = self.dummies or []

    def _set_class_name(self, name: str) -> None:
        """Set the class name for this index.

        Called by the Generators container when the index is assigned to an attribute.
        """
        self.class_name = name

    def get_class_name(self) -> str:
        """Return the class name, or None if not set."""
        return self.class_name

    def generate_with_context(
        self, user_data_types: List[str] = None
    ) -> str:  # noqa: ARG002
        """Generate the C++ source code for the custom index class."""
        return _generate_index(
            self.class_name, self.dims, self.strides, self.dummies, self.init
        )

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
    return '#include "spio/compound_index.h"'


def _generate_index(
    class_name: str,
    dims: Dims,
    strides: Strides,
    dummy_dims: List[str] = None,
    init: BuiltIn = None,
) -> str:
    """Generate a using statement for an CompoundIndex template instantiation."""
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

        if dim_class in BUILTIN_DIM_NAMES:
            dim_class = "spio::" + dim_class

        # Add the DimInfo parameter
        dim_infos.append(f"spio::DimInfo<{dim_class}, {size_str}, {stride}>")

        # If this is a dummy dimension, generate a specialization
        if name in (dummy_dims or []):
            dim_info = f"spio::DimInfo<{dim_class}, {size_str}, {stride}>"
            specializations.append(
                f"namespace spio {{ namespace detail {{\n"
                f"    template<> struct is_dummy_dimension<{dim_info}> {{\n"
                f"        static constexpr bool value = true;\n"
                f"    }};\n"
                f"}}}}\n"
            )

    # Generate the base type
    base_type = f"spio::CompoundIndex<{', '.join(dim_infos)}>"

    if init is None:
        # Generate a simple using statement
        index_code = f"using {class_name} = {base_type};\n"
    else:
        # Generate a derived struct with an initializing constructor
        index_code = (
            f"struct {class_name} : {base_type} {{\n"
            f"    {class_name}() : {base_type}({init.value}) {{}}\n"
            f"}};\n"
        )

    # Combine the code with any specializations
    return index_code + "\n".join(specializations)
