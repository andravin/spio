"""Code generator for custom tensor classes in CUDA source code."""

from typing import Dict, Tuple
from dataclasses import dataclass
from math import prod

from .index import _generate_index

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
class TensorSpec:
    """CUDA Code generator for custom tensor classes.

    This class is used to generate custom tensor classes that map named tensor
    dimensions to pointers.

    Attributes:
        class_name (str): The name of the custom tensor class.
        data_type (str): The data type of the tensor elements (e.g., float, uint4).
        dims (Dict[str, int]): A dictionary mapping dimension names to their sizes.
    """

    class_name: str
    data_type: str
    dims: Dict[str, int]

    def generate(self) -> str:
        """Generate the C++ source code for the custom tensor class."""
        return _generate_tensor(self.class_name, self.data_type, self.dims)

    @property
    def num_bytes(self) -> int:
        """The number of bytes required to store the tensor data."""
        return prod(self.dims.values()) * _sizeof_data_type(self.data_type)


def _generate_tensor(class_name: str, data_type: str, dims: Dict[str, int]) -> str:
    """Return the C++ source code that implements a custom tensor class.

    Custom tensor classes use named tensor dimensions.

    Parameters:
        class_name(str): the name to use for the C++ class
        data_type(str): the C++/CUDA data-type used by the Tensor elements.
        dims(Dict[str, int]): an (ordered) dict that maps dimension names to their sizes.
    """
    code = ""
    index_class_name = f"_{class_name}_Index"
    code += _generate_index(index_class_name, dims)
    code += "\n"
    code += _class(class_name, data_type, tuple(dims.values()))
    code += _using_index(index_class_name)
    for name, value in dims.items():
        code += _dim(name, value)
    for d, name in enumerate(dims.keys()):
        code += _dim_to_pointer(class_name, d, name)
    code += _tail()
    return code


def _tensor_header():
    """The C++ statement that includes the spio tensor header file.

    This file implements the C++ base template classes from which the
    custom tensor classes inherit. You must include this header before
    using the code returned by the generate_tensor() function.
    """
    return '#include "spio/tensor.h"'


def _using_index(index_class_name: str):
    return f"""
        using Index = {index_class_name};

        static constexpr int size = Index::size;

        static constexpr int num_bytes = size * sizeof(data_type);
"""


def _class(class_name: str, data_type: str, shape: Tuple[int, ...]) -> str:
    num_dims = len(shape)
    shape_str = ", ".join([str(d) for d in shape[1:]])
    base = f"Tensor{num_dims}D<{data_type}, {shape_str}>"
    return f"""
    class {class_name} : public spio::{base} {{
    public:
        using Base = {base};

        using data_type = {data_type};

        DEVICE constexpr {class_name}({data_type} *data  = nullptr) : Base(data) {{}}

        DEVICE constexpr {class_name}(const {base} &other) : Base(other) {{}}

        DEVICE const {class_name}& operator=(const {base} &other) {{ this->reset(other.get()); return *this; }}
 """


def _dim(name: str, value: int) -> str:
    name = name.upper()
    return f"""
        constexpr static int {name} = {value};
    """


def _dim_to_pointer(class_name: str, d: int, name: str) -> str:
    name_in = f"{name}_in"
    dim_d = f"_d{d}"
    return f"""
        DEVICE constexpr {class_name} {name}(int {name_in}) const {{ return {dim_d}({name_in}); }}
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
