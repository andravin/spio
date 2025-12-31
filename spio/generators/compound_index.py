"""Code generator for custom index classes in CUDA / C++."""

from typing import List, Generator
from dataclasses import dataclass

from .dims import Dims, Strides, compute_full_strides
from .dim import dim_name_to_dim_or_fold_class_name, BUILTIN_DIM_NAMES
from .built_in import BuiltIn
from .gen_specs import GenSpecsWithContext, GenSpecs
from .derived_dimension import DerivedDimension


@dataclass
class CompoundIndex(GenSpecsWithContext, DerivedDimension):
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

    def partition(
        self,
        partition_dim: str,
        partition_index: "CompoundIndex",
    ) -> "CompoundIndexPartition":
        """Create a CompoundIndexPartition generator for this index.

        CompoundIndexPartition generates a range function for iterating over
        the elements of a compound index by multiple threads in parallel.
        Each thread handles elements at offsets: start, start+stride, start+2*stride, ...
        where start comes from partition_index and stride is the size of partition_dim.

        Args:
            partition_dim: The dimension to partition by (e.g., "LANE"). The size of
                this dimension in partition_index determines the iteration stride.
            partition_index: Provides the starting offset via get<partition_dim>().
                Typically a thread index like ComputeIndex with a LANE dimension.

        Returns:
            CompoundIndexPartition: The partitioned index generator.

        Example:
            g.CLoadSmemIndex = CompoundIndex(Dims(i=32, j8=8))
            g.ComputeIndex = CompoundIndex(Dims(warp=4, lane=32), init=BuiltIn.THREAD_IDX_X)
            g.partition_c_smem = g.CLoadSmemIndex.partition("lane", g.ComputeIndex)

            # Generates: auto partition_c_smem() { return CLoadSmemIndex::partition<LANE>(ComputeIndex()); }
            # In kernel: for (auto p : partition_c_smem()) { ... }
        """
        return CompoundIndexPartition(
            base_index=self,
            partition_dim=partition_dim,
            partition_index=partition_index,
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


@dataclass
class CompoundIndexPartition(GenSpecs):
    """CUDA Code generator for partitioned iteration over a compound index.

    Generates a function that returns a range for cooperative iteration where
    multiple threads process elements in parallel. Each thread starts at a
    different offset (determined by partition_index) and strides by the size
    of partition_dim.

    When used with the Generators container, function_name can be omitted and
    will be set from the attribute name.

    Attributes:
        base_index: The CompoundIndex whose elements are iterated over.
        partition_dim: The dimension name for partitioning (e.g., "LANE").
        partition_index: Provides starting offset and stride size for partition_dim.
        function_name: The generated function name (optional with Generators).
    """

    base_index: CompoundIndex
    partition_dim: str
    partition_index: CompoundIndex
    function_name: str = None

    def _set_class_name(self, name: str) -> None:
        """Set the class name for this index.

        Called by the Generators container when the index is assigned to an attribute.
        """
        self.function_name = name

    def get_class_name(self) -> str:
        """Return the class name, or None if not set."""
        return self.function_name

    def used_generators(self) -> list[GenSpecsWithContext]:
        """Return the list of generators used by this generator."""
        return [self.base_index, self.partition_index]

    def generate(self) -> str:
        """Generate the C++ source code for the custom partitioned index class."""
        if self.function_name is None:
            raise ValueError("CompoundIndexPartition requires a class _name")

        index_class_name = self.base_index.get_class_name()
        if index_class_name is None:
            raise ValueError(
                "CompoundIndexPartition requires a base_index with a function_name"
            )

        partition_index_class_name = self.partition_index.get_class_name()
        if partition_index_class_name is None:
            raise ValueError(
                "CompoundIndexPartition requires a partition_index with a function_name"
            )

        partition_dim = dim_name_to_dim_or_fold_class_name(self.partition_dim.upper())
        if partition_dim in BUILTIN_DIM_NAMES:
            partition_dim = "spio::" + partition_dim

        return f"""
auto {self.function_name}() {{ return {index_class_name}::partition<{partition_dim}>({partition_index_class_name}()); }}
"""
