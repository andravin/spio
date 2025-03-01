"""Code generator for custom tensor classes in CUDA source code."""

from typing import Dict, Tuple, Union, Generator
from dataclasses import dataclass

from .index import _generate_index
from .subindex_protocol import SubindexProtocol
from .dim import dim_name_to_dim_or_fold_class_name
from .dims import Dims

DATA_TYPE_SIZES = {
    "float": 4,
    "float2": 8,
    "float4": 16,
    "unsigned": 4,
    "uint2": 8,
    "uint4": 16,
    "half": 2,
    "half2": 4,
}


@dataclass
class Tensor:
    """CUDA Code generator for custom tensor classes.

    This class is used to generate custom tensor classes that map named tensor
    dimensions to pointers.

    The user may optionally optionally set the stride of a dimension
    by specifying it in the strides parameter. Any unspecified stride is automatically
    set to the size of the next dimension times the stride of the next dimension.

    Attributes:
        class_name (str): The name of the custom tensor class.
        data_type (str): The data type of the tensor elements (e.g., float, uint4).
        dims (Dict[str, int]): A dictionary mapping dimension names to their sizes.
        strides (Dict[str, int]): An optional dictionary mapping dimension names to their strides.
    """

    class_name: str
    data_type: str
    dims: Dims
    strides: Dict[str, int] = None

    def __post_init__(self):
        self.strides = self._compute_full_strides(self.dims, self.strides)

    @classmethod
    def _compute_full_strides(
        cls,
        dims: Dict[str, Union[int, SubindexProtocol]],
        given_strides: Dict[str, int],
    ) -> Dict[str, int]:
        """Compute the full strides for the given dimensions."""
        if given_strides is None:
            given_strides = {}
        all_strides = {}
        stride = 1
        for name, value in reversed(dims.items()):
            if isinstance(value, SubindexProtocol):
                value = value.size
            if name in given_strides:
                stride = given_strides[name]
            all_strides[name] = stride
            # Compute the default stride of the next dimension.
            stride *= value
        return all_strides

    def generate(self) -> str:
        """Generate the C++ source code for the custom tensor class."""
        return _generate_tensor(
            self.class_name, self.data_type, self.dims, self.strides, self.size
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
        return self.size * _sizeof_data_type(self.data_type)

    @property
    def dim_names(self) -> Generator[str, None, None]:
        """Return the names of the dimensions in the tensor."""
        for name, value in self.dims.items():
            if isinstance(value, SubindexProtocol):
                yield from value.dim_names
            else:
                yield name


def _generate_tensor(
    class_name: str,
    data_type: str,
    dims: Dict[str, int],
    strides: Dict[str, int],
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
        data_type(str): the C++/CUDA data-type used by the Tensor elements.
        dims(Dict[str, int]): an (ordered) dict that maps dimension names to their sizes.
        strides(Dict[str, int]): optional dict that specifies non-default strides for given dims.
    """
    code = ""
    index_class_name = f"_{class_name}_Index"
    code += _generate_index(index_class_name, dims)
    code += "\n"
    sizes = _sizes_gen(dims)
    code += _class(class_name, data_type, tuple(sizes), tuple(strides.values()))
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
    class_name: str, data_type: str, shape: Tuple[int, ...], stride: Tuple[int, ...]
) -> str:
    num_dims = len(shape)
    shape_str = ", ".join([str(d) for d in shape[1:]])
    stride_str = ", ".join([str(d) for d in stride])
    if shape_str and stride_str:
        template_pars_str = f"<{data_type}, {shape_str}, {stride_str}>"
    else:
        assert (
            stride_str == "1"
        ), f"Unexpected template parameters: shape:{shape_str} stride:{stride_str}"
        template_pars_str = f"<{data_type}>"
    base = f"Tensor{num_dims}D{template_pars_str}"
    return f"""
    class {class_name} : public spio::{base} {{
    public:
        using Base = {base};

        DEVICE constexpr {class_name}({data_type} *data  = nullptr) : Base(data) {{}}

        DEVICE constexpr {class_name}(const {base} &other) : Base(other) {{}}

        DEVICE const {class_name}& operator=(const {base} &other) {{ this->reset(other.get()); return *this; }}
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


def _sizeof_data_type(data_type: str) -> int:
    return DATA_TYPE_SIZES[_strip_data_type(data_type)]


def _strip_data_type(data_type: str) -> str:
    if data_type.startswith("const "):
        return _strip_data_type(data_type[6:])
    if data_type.startswith("__half"):
        return _strip_data_type("half" + data_type[6:])
    return data_type
