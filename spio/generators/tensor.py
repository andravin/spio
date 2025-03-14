"""Code generator for custom tensor classes in CUDA source code."""

from typing import Dict, Tuple, Union, Generator, List
from dataclasses import dataclass

from .index import _generate_index
from .subindex_protocol import SubindexProtocol
from .dim import dim_name_to_dim_or_fold_class_name
from .dims import Dims, Strides, compute_full_strides
from .fragment_type import FragmentType
from .data_type import dtype


@dataclass
class Tensor:
    """CUDA Code generator for custom tensor classes.

    This class is used to generate custom tensor classes that map named tensor
    dimensions to pointers.

    The user may optionally optionally set the stride of a dimension
    by specifying it in the strides parameter. Any unspecified stride is automatically
    calculated as the the size of the next dimension times the stride of the next dimension,
    with the last dimension having a stride of 1.

    Attributes:
        class_name (str): The name of the custom tensor class.
        data_type (Union[dtype, FragmentType]): The data type of the tensor elements.
        dims (Dims): A dictionary mapping dimension names to their sizes.
        strides (Strides): An optional dictionary mapping dimension names to their strides.
    """

    class_name: str
    data_type: Union[dtype, FragmentType]
    dims: Dims
    strides: Strides = None
    constant: bool = False

    def __post_init__(self):
        self.strides = compute_full_strides(self.dims, self.strides)

    def generate_with_context(self, user_types: List[str] = None) -> str:
        """Generate the C++ source code for the custom tensor class."""
        data_type_name = self._get_data_type_name(user_types=user_types)
        return _generate_tensor(
            self.class_name,
            data_type_name,
            self.dims,
            self.strides,
            self.size,
        )

    @property
    def size(self) -> int:
        """The number of elements required to store the tensor data.

        This includes any padding introduced by strides that are greater
        than the corresponding dimension sizes.
        """
        name_0, size_0 = next(iter(self.dims.items()))
        if isinstance(size_0, SubindexProtocol):
            size_0 = size_0.size
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
        for name, value in self.dims.items():
            if isinstance(value, SubindexProtocol):
                yield from value.dim_names
            else:
                yield name

    def _get_data_type_name(self, user_types: List[str]) -> str:
        """Return the type-name for the tensor data type.

        The type-name is the literal name of the data type used in CUDA / C++ code.
        """
        return _get_data_type_name(
            self.data_type, constant=self.constant, user_types=user_types
        )


def _generate_tensor(
    class_name: str,
    data_type_name: str,
    dims: Dims,
    strides: Strides,
    size: int = None,
) -> str:
    """Return the C++ source code that implements a custom tensor class.

    Custom tensor classes use named tensor dimensions. The tensor class encapsulates
    a pointer to the given data_type and provides methods to that return a new Tensor
    object that point to a specific element at a given index. You can chain accessors
    together like this:

            tensor.z(3).y(2).x(1)

    Would return a Tensor object that points to the element at indices (x=1, y=2, z=3).

    The tensor class includes an Index class that performs the inverse mapping that folds
    a composite index into the individual indices for each dimension:

             # Use the thread index as a compound index that selects a tensor element.
             Tensor::Index idx(threadIdx.x);

             # Get the coordinates of the tensor element.
             int x = idx.x();
             int y = idx.y();
             int z = idx.z();

    The default dimension strides are computed as the product of the sizes of the trailing
    dimensions. The user can specify a non-default stride for any dimension using the strides
    parameter:

             # Increase the natural stride of the y-dimension to 5 (default is 4).
             TensorSpec("Tensor", "float", {"x": 4, "y": 4, "z": 4}, strides={"y": 5})

    The strides only affect the pointers calculated by the index accessors. The Tensor::Index
    class still uses the default strides to compute the individual indices from the compound index.
    This allows a contiguous compound index to be folded into tensor indices that can be used
    to access elements of a non-contiguous tensor.

    Parameters:
        class_name(str): the name to use for the C++ class
        data_type_name(str): the C++/CUDA data-type used by the Tensor elements.
        dims(Dims): a Dims object that defines the dimension names and sizes.
        strides(Dict[str, int]): optional dict that specifies non-default strides for given dims.
    """
    code = ""
    index_class_name = f"_{class_name}_Index"
    code += _generate_index(index_class_name, dims)
    code += "\n"
    sizes = _sizes_gen(dims)
    code += _class(
        class_name,
        data_type_name,
        tuple(sizes),
        tuple(strides.values()),
    )
    code += _using_index(index_class_name)
    code += _size_and_bytes(size)
    for (name, value), size in zip(dims.items(), sizes):
        code += _dim(name, value, size)
    for name in dims.keys():
        code += _stride(name, strides[name])
    for d, (name, value) in enumerate(dims.items()):
        code += _dim_to_pointer(class_name, d, name, value)
    code += _tail()
    return code


def _sizes_gen(dims: Dict[str, int]):
    return [d.size if isinstance(d, SubindexProtocol) else d for d in dims.values()]


def header():
    """The C++ statement that includes the spio tensor header file.

    This file implements the C++ base template classes from which the
    custom tensor classes inherit. You must include this header before
    using the code returned by the generate_tensor() function.
    """
    return '#include "spio/tensor.h"'


def _using_index(index_class_name: str):
    return f"""
        using Index = {index_class_name};
"""


def _size_and_bytes(size: int) -> str:
    return f"""
        static constexpr int size = {size};

        static constexpr int num_bytes = size * sizeof(data_type);
"""


def _class(
    class_name: str,
    data_type_name: str,
    shape: Tuple[int, ...],
    stride: Tuple[int, ...],
) -> str:
    num_dims = len(shape)
    shape_str = ", ".join([str(d) for d in shape[1:]])
    stride_str = ", ".join([str(d) for d in stride])
    if shape_str and stride_str:
        template_pars_str = f"<{data_type_name}, {shape_str}, {stride_str}>"
    else:
        assert (
            stride_str == "1"
        ), f"Unexpected template parameters: shape:{shape_str} stride:{stride_str}"
        template_pars_str = f"<{data_type_name}>"
    base = f"Tensor{num_dims}D{template_pars_str}"
    return f"""
    class {class_name} : public spio::{base} {{
    public:
        using Base = {base};

        DEVICE constexpr {class_name}({data_type_name} *data  = nullptr) : Base(data) {{}}

        DEVICE constexpr {class_name}(const {base} &other) : Base(other) {{}}

        DEVICE const {class_name}& operator=(const {base} &other) {{ this->reset(other.get()); return *this; }}

        DEVICE constexpr {class_name} offset(int idx) const {{ return {class_name}(this->get() + idx); }}

        template<typename IndexType>
        DEVICE constexpr {class_name} operator[](IndexType idx) const {{ return {class_name}(idx.offset_tensor(*this)); }}

        DEVICE static {class_name} allocate(spio::StackAllocator &allocator) {{           
            return {class_name}(allocator.allocate<data_type>(size));
        }}

        DEVICE void deallocate(spio::StackAllocator &allocator) {{
            allocator.deallocate(this->get(), size);
        }}
 """


def _dim(name: str, value: Union[int, SubindexProtocol], size: int) -> str:
    name = name.upper()
    dim_type = (
        "int"
        if isinstance(value, SubindexProtocol)
        else dim_name_to_dim_or_fold_class_name(name)
    )
    return f"""
        static constexpr {dim_type} {name} = {size};
"""


def _stride(name: str, value: int) -> str:
    name = name.upper()
    return f"""
        static constexpr int {name}_Stride = {value};
    """


def _dim_to_pointer(
    class_name: str, d: int, name: str, value: Union[int, SubindexProtocol]
) -> str:
    name_in = f"{name}_in"
    dim_class_name = dim_name_to_dim_or_fold_class_name(name)
    dim_d = f"_d{d}"
    if isinstance(value, SubindexProtocol):
        decl_str = value.generate_offset_function_declaration(
            return_type=class_name, function_name=name
        )
        call_str = value.generate_offset_function_call()
        return f"""
        {decl_str}
            const int compound_index = {call_str};
            return {dim_d}(compound_index);
        }}
"""
    else:
        return f"""
        /// Return a {dim_class_name} object that points to the element at the given index.
        DEVICE constexpr {class_name} {name}({dim_class_name} {name_in}) const {{ return {dim_d}({name_in}.get()); }}

        /// Return a {dim_class_name} object that points to the element at the given index.
        /// This overloads the subscript operator "[]".
        DEVICE constexpr {class_name} operator[]({dim_class_name} {name_in}) const {{ return {dim_d}({name_in}.get()); }}
"""


def _tail() -> str:
    return """
    };
"""


def _get_data_type_name(
    data_type: Union[dtype, FragmentType],
    constant: bool = False,
    user_types: List[str] = None,
) -> str:
    if isinstance(data_type, FragmentType):
        data_type = f"spio::{data_type.value}"
    elif isinstance(data_type, dtype):
        data_type = data_type.value.name
    elif isinstance(data_type, str):
        if user_types is None:
            raise ValueError("user_types must be provided for user-defined data-types.")
        if not data_type in user_types:
            raise ValueError(f"Unknown user-defined data-type: {data_type}")
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
    if constant:
        data_type = f"const {data_type}"
    return data_type
