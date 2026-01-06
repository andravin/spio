"""Code generator for custom tensor classes in CUDA source code."""

from typing import Dict, Union, Generator, List, Tuple
from dataclasses import dataclass, field

from .dim import dim_name_to_dim_or_fold_class_name
from .dims import Dims, Strides, compute_full_strides
from .fragment_type import FragmentType
from .fragment import Fragment
from .data_type import dtype, get_dtype_veclen, get_dtype_with_veclen
from .gen_specs import GenSpecs, GenSpecsWithContext
from .derived_dimension import DerivedDimension, SizedDerivedDimension


@dataclass
class Tensor(GenSpecsWithContext):
    """CUDA Code generator for custom tensor classes.

    This class is used to generate custom tensor classes that map named tensor
    dimensions to pointers.

    The user may optionally set the stride of a dimension by specifying it in
    the strides parameter. Any unspecified stride is automatically calculated
    as the size of the next dimension times the stride of the next dimension,
    with the last dimension having a stride of 1.

    When used with the Generators container, class_name can be omitted and will
    be set from the attribute name.

    Attributes:
        data_type (Union[dtype, FragmentType, str]): The data type of the tensor elements.
        dims (Dims): A dictionary mapping dimension names to their sizes.
        class_name (str): The name of the custom tensor class (optional with Generators).
        strides (Strides): An optional dictionary mapping dimension names to their strides.
        constant (bool): Whether the tensor is constant.
        derived_dims (List[DerivedDimension]): List of derived dimensions used in dims.
        ancestors (List["Tensor"]): List of tensors from which this tensor is derived.
    """

    data_type: Union[dtype, FragmentType, str]
    dims: Dims
    class_name: str = None
    strides: Strides = None
    constant: bool = False
    derived_dims: List[DerivedDimension] = None
    ancestors: List["Tensor"] = None

    def __post_init__(self):
        if isinstance(self.dims, dict):
            self.dims = Dims(**self.dims)
        if isinstance(self.strides, dict):
            self.strides = Strides(**self.strides)
        if self.strides is not None:
            for stride_name in self.strides.keys():
                if stride_name not in self.dims:
                    raise ValueError(
                        f"Stride name '{stride_name}' not found in dims {list(self.dims.keys())}."
                    )
        for name, value in self.dims.items():
            if isinstance(value, SizedDerivedDimension):
                value.set_output_dim_name(name)
        self.strides = compute_full_strides(self.dims, self.strides)

    def _set_class_name(self, name: str) -> None:
        """Set the class name for this tensor.

        Called by the Generators container when the tensor is assigned to an attribute.
        """
        self.class_name = name

    def get_class_name(self) -> str:
        """Return the class name, or None if not set."""
        return self.class_name

    def generate_with_context(self, user_data_types: List[str] = None) -> str:
        """Generate the C++ source code for the custom tensor class."""
        data_type_name = self._get_data_type_name(user_data_types=user_data_types)
        return _generate_tensor(
            self.class_name,
            data_type_name,
            self.dims,
            self.strides,
            derived_dims=self.derived_dims,
        )

    def used_generators(self) -> list[GenSpecs]:
        """Return a list of generator class-names used by this generator.

        This is used to find any generators that need to be assigned class names automatically.
        Subclasses should override this method if they use other generators.
        """
        used_gens = []
        for dim_value in self.dims.values():
            if isinstance(dim_value, DerivedDimension):
                used_gens.append(dim_value)
        if self.derived_dims:
            used_gens.extend(self.derived_dims)
        return used_gens

    def with_dim(self, derived_dim: DerivedDimension) -> "Tensor":
        """Return a new Tensor with an additional derived dimension.

        Duplicate this tensor and add the given derived dimension to its derived_dims list.
        """
        if not isinstance(derived_dim, DerivedDimension):
            raise ValueError(
                f"with_dim requires a DerivedDimension, got {type(derived_dim)}"
            )
        new_derived_dims = list(self.derived_dims) if self.derived_dims else []
        new_derived_dims.append(derived_dim)
        new_ancestors = list(self.ancestors) if self.ancestors else []
        new_ancestors.append(self)
        return Tensor(
            data_type=self.data_type,
            dims=self.dims,
            class_name=None,
            strides=self.strides,
            constant=self.constant,
            derived_dims=new_derived_dims,
            ancestors=new_ancestors,
        )

    def vector_length(self, width: int, constant: bool = None) -> "Tensor":
        """Return a new Tensor with a different vector width.

        This method creates a new tensor with a wider vector type, adjusting
        the dimension with stride=1 to account for the increased vector size.

        Args:
            width: The new vector width (e.g., 8 for half8).
            constant: Whether the new tensor is constant. If None, inherits from self.

        Returns:
            A new Tensor with the adjusted dtype, dims, and strides.

        Raises:
            ValueError: If the width ratio does not divide the stride-1 dimension's size.

        Example:
            Given a tensor with dtype.half2 and dims (warp_i, j8, i, j2=4),
            calling with_vector_width(8) returns a tensor with dtype.half8
            and dims (warp_i, j8, i), with strides divided by 4.
        """
        if not isinstance(self.data_type, dtype):
            raise ValueError(
                f"with_vector_width requires a dtype, got {self.data_type}"
            )

        current_veclen = get_dtype_veclen(self.data_type)
        if width <= current_veclen:
            raise ValueError(
                f"New width {width} must be greater than current vector length {current_veclen}"
            )

        if width % current_veclen != 0:
            raise ValueError(
                f"New width {width} must be a multiple of current vector length {current_veclen}"
            )

        width_ratio = width // current_veclen

        # Find the dimension with stride=1
        stride1_dim = None
        for name, stride in self.strides.items():
            if stride == 1:
                stride1_dim = name
                break

        if stride1_dim is None:
            raise ValueError("No dimension with stride=1 found")

        stride1_size = self.dims[stride1_dim]
        if stride1_size % width_ratio != 0:
            raise ValueError(
                f"Width ratio {width_ratio} must divide the size of dimension "
                f"'{stride1_dim}' ({stride1_size})"
            )

        # Build new dims and strides
        new_dims_dict = {}
        new_strides_dict = {}

        for name, size in self.dims.items():
            if name == stride1_dim:
                new_size = size // width_ratio
                if new_size > 1:
                    # Reduce dimension size
                    new_dims_dict[name] = new_size
                    new_strides_dict[name] = 1
                # If new_size == 1, eliminate the dimension entirely
            else:
                new_dims_dict[name] = size
                # Divide stride by width_ratio
                new_strides_dict[name] = self.strides[name] // width_ratio

        new_dims = Dims(**new_dims_dict)
        new_strides = Strides(**new_strides_dict) if new_strides_dict else None

        # Get the new dtype with the target vector length
        new_dtype = get_dtype_with_veclen(self.data_type, width)

        new_ancestors = list(self.ancestors) if self.ancestors else []
        new_ancestors.append(self)
        return Tensor(
            data_type=new_dtype,
            dims=new_dims,
            class_name=None,
            strides=new_strides,
            constant=self.constant if constant is None else constant,
            derived_dims=self.derived_dims,
            ancestors=new_ancestors,
        )

    def initializer(self, *implicit_dims: GenSpecsWithContext) -> "CursorInitializer":
        """Return a CursorInitializer that applies implicit subscripts at construction.

        Implicit dimensions are subscripts that are automatically applied when
        the cursor is created, using default-constructed index types. This is
        useful for absorbing thread-specific indexing (e.g., based on THREAD_IDX_X)
        into the cursor factory.

        Args:
            *implicit_dims: Generator specs for the implicit dimension types.
                Each must be a generator with a class_name that will be
                default-constructed and used as a subscript.

        Returns:
            A CursorInitializer generator that produces a factory function.

        Example:
            g.AGlobalLoader = g.AGlobal.implicit_dim(g.ALoadGlobalIndex)

            Generates:
            auto AGlobalLoader(const half* ptr) {
                return AGlobal(ptr)[ALoadGlobalIndex()];
            }
        """
        return CursorInitializer(tensor=self, implicit_dims=list(implicit_dims))

    def __getitem__(
        self, implicit_dims: Union[GenSpecsWithContext, Tuple[GenSpecsWithContext, ...]]
    ) -> "CursorInitializer":
        """Subscript operator as a synonym for initializer.

        Allows using tensor[dim] or tensor[dim1, dim2] syntax instead of
        tensor.initializer(dim) or tensor.initializer(dim1, dim2).

        Args:
            implicit_dims: A single generator spec or tuple of generator specs
                for the implicit dimension types.

        Returns:
            A CursorInitializer generator that produces a factory function.

        Example:
            g.AGlobalLoader = g.AGlobal[g.ALoadGlobalIndex]
            # Equivalent to: g.AGlobalLoader = g.AGlobal.initializer(g.ALoadGlobalIndex)
        """
        if isinstance(implicit_dims, tuple):
            return self.initializer(*implicit_dims)
        return self.initializer(implicit_dims)

    @property
    def size(self) -> int:
        """The number of elements required to store the tensor data."""
        name_0, size_0 = next(iter(self.dims.items()))
        stride_0 = self.strides[name_0]
        return size_0 * stride_0

    @property
    def num_bytes(self) -> int:
        """The number of bytes required to store the tensor data."""
        if isinstance(self.data_type, dtype):
            element_size = self.data_type.value.size
        else:
            raise ValueError(f"Size of data_type {self.data_type} not supported.")

        return self.size * element_size

    @property
    def dim_names(self) -> Generator[str, None, None]:
        """Return the names of the dimensions in the tensor."""
        for name, _ in self.dims.items():
            yield name

    def _get_data_type_name(self, user_data_types: List[str]) -> str:
        """Return the type-name for the tensor data type.

        The type-name is the literal name of the data type used in CUDA / C++ code.
        """
        return _get_data_type_name(
            self.data_type, constant=self.constant, user_data_types=user_data_types
        )


@dataclass
class CursorInitializer(GenSpecsWithContext):
    """Code generator for a cursor factory with implicit dimension subscripts.

    This class generates a factory function that constructs a tensor cursor
    and applies default-constructed subscripts for each implicit dimension.
    Implicit dimensions are evaluated once when the factory is called, making
    them ideal for thread-specific indexing (e.g., using THREAD_IDX_X).

    Unlike derived dimensions (which affect template parameters and match
    during subscripting), implicit dimensions are simply subscripts applied
    at cursor construction time.

    Attributes:
        tensor: The base Tensor to create cursors from.
        implicit_dims: List of generators for the implicit dimension types.
        class_name: The name of the factory function (optional with Generators).

    Example:
        g.AGlobalLoader = g.AGlobal.implicit_dim(g.ALoadGlobalIndex)

        Generates:
        auto AGlobalLoader(const half* ptr) {
            return AGlobal(ptr)[ALoadGlobalIndex()];
        }
    """

    tensor: "Tensor"
    implicit_dims: List[GenSpecsWithContext] = field(default_factory=list)
    class_name: str = None

    def _set_class_name(self, name: str) -> None:
        """Set the function name for this cursor factory.

        Called by the Generators container when assigned to an attribute.
        """
        self.class_name = name

    def get_class_name(self) -> str:
        """Return the function name, or None if not set."""
        return self.class_name

    def used_generators(self) -> list[GenSpecsWithContext]:
        """Return the tensor and implicit dimension generators."""
        used = [self.tensor] + list(self.implicit_dims)
        for dim in self.implicit_dims:
            if hasattr(dim, "used_generators"):
                used.extend(dim.used_generators())
        return used

    def generate_with_context(self, user_data_types: List[str] = None) -> str:
        """Generate the C++ cursor subclass.

        Generates a struct that:
        - Inherits from the tensor's cursor_type
        - Applies implicit subscripts at construction time
        - Inherits all cursor methods (rebase, operator[], inbounds, get, etc.)
        """
        if self.class_name is None:
            raise ValueError("CursorInitializer requires a class_name")

        tensor_class_name = self.tensor.get_class_name()
        if tensor_class_name is None:
            raise ValueError("Tensor must have a class_name set")

        # Build the subscript chain
        subscript_chain = ""
        for implicit_dim in self.implicit_dims:
            dim_class_name = implicit_dim.get_class_name()
            if dim_class_name is None:
                raise ValueError("Implicit dimension must have a class_name set")
            subscript_chain += f"[{dim_class_name}()]"

        # Generate constructor overloads for ancestor tensor types
        ancestor_constructors = ""
        if self.tensor.ancestors:
            for ancestor in self.tensor.ancestors:
                ancestor_class_name = ancestor.get_class_name()
                if ancestor_class_name is None:
                    raise ValueError("Ancestor tensor must have a class_name set")
                ancestor_constructors += (
                    f"    DEVICE {self.class_name}({ancestor_class_name} tensor)\n"
                    f"        : Base({tensor_class_name}("
                    f"reinterpret_cast<data_type*>(tensor.get())){subscript_chain}) {{}}\n"
                )

        return (
            f"struct {self.class_name} : {tensor_class_name}::cursor_type {{\n"
            f"    using Base = typename {tensor_class_name}::cursor_type;\n"
            f"    using data_type = typename Base::data_type;\n"
            f"    DEVICE {self.class_name}(data_type* ptr)\n"
            f"        : Base({tensor_class_name}(ptr){subscript_chain}) {{}}\n"
            f"    DEVICE {self.class_name}({tensor_class_name} tensor)\n"
            f"        : Base(tensor{subscript_chain}) {{}}\n"
            f"{ancestor_constructors}"
            f"}};\n"
        )

    @property
    def strides(self) -> Strides:
        """Return the strides of the underlying tensor."""
        return self.tensor.strides


def _generate_tensor(
    class_name: str,
    data_type_name: str,
    dims: Dims,
    strides: Dict[str, int],
    derived_dims: List[DerivedDimension] = None,
) -> str:
    """Generate a using statement for a Tensor template instantiation."""

    # Make a shallow copy of the derived_dims argument to avoid modifying it.
    derived_dims = list(derived_dims) if derived_dims else []

    dim_infos = []
    for name, dim_value in dims.items():
        if isinstance(dim_value, SizedDerivedDimension):
            size = dim_value.size
            derived_dims.append(dim_value)
        else:
            size = dim_value
        dim_class = dim_name_to_dim_or_fold_class_name(name)
        size_str = str(size)
        stride = strides[name]
        dim_infos.append(f"spio::DimInfo<{dim_class}, {size_str}, {stride}>")

    # Add derived dimension class names after all DimInfo entries
    for derived_dim in derived_dims:
        dim_infos.append(derived_dim.get_class_name())

    # More concise formatting with line breaks only for longer declarations
    if len(dim_infos) <= 3:
        dim_info_str = ", ".join(dim_infos)
        return f"using {class_name} = spio::Tensor<{data_type_name}, {dim_info_str}>;\n"
    else:
        dim_info_str = ",\n    ".join(dim_infos)
        return f"using {class_name} = spio::Tensor<{data_type_name},\n    {dim_info_str}\n>;\n"


def header():
    """The C++ statement that includes the spio tensor header file.

    This file implements the C++ base template classes from which the
    custom tensor classes inherit. You must include this header before
    using the code returned by the generate_tensor() function.
    """
    return '#include "spio/tensor.h"'


def _get_data_type_name(
    data_type: Union[dtype, FragmentType],
    constant: bool = False,
    user_data_types: List[str] = None,
) -> str:
    if isinstance(data_type, FragmentType):
        data_type = f"spio::{data_type.value}"
    elif isinstance(data_type, Fragment):
        data_type = data_type.class_name
    elif isinstance(data_type, dtype):
        data_type = data_type.value.name
    elif isinstance(data_type, str):
        if user_data_types is None:
            raise ValueError(
                "user_data_types must be provided for user-defined data-types."
            )
        if not data_type in user_data_types:
            raise ValueError(f"Unknown user-defined data-type: {data_type}")
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
    if constant:
        data_type = f"const {data_type}"
    return data_type
