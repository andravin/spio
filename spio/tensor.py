from typing import Dict, Tuple, List
from dataclasses import dataclass


def tensor_header():
    """Return the C++ statement that includes the spio tensor header file.

    This file implements the C++ base template classes from which the
    custom tensor classes inherit. You must include this header before using
    the code returned by the generate_tensor() function.
    """
    return '#include "spio/tensor.h"'


@dataclass
class TensorSpec:
    class_name: str
    data_type: str
    dims: Dict[str, int]


def generate_tensors(tensor_specs: List[TensorSpec], header_file_name: str) -> None:
    code = tensor_header()
    code += ""
    for tensor_spec in tensor_specs:
        code += generate_tensor(
            tensor_spec.class_name, tensor_spec.data_type, tensor_spec.dims
        )
        code += ""

    with open(header_file_name, "w") as file:
        file.write(code)


def generate_tensor(class_name: str, data_type: str, dims: Dict[str, int]) -> str:
    """Return the C++ source code that implements a custom tensor class.

    Custom tensor classes use named tensor dimensions.

    Parameters:
        class_name(str): the name to use for the C++ class
        data_type(str): the C++/CUDA data-type used by the Tensor elements.
        dims(Dict[str, int]): an (ordered) dict that maps dimension names to their sizes.
    """
    code = ""
    code += _class(class_name, data_type, tuple(dims.values()))
    for name, value in dims.items():
        code += _dim(name, value)
    for d, name in enumerate(dims.keys()):
        code += _dim_to_pointer(class_name, d, name)
    code += _tail()
    return code


def _class(class_name: str, data_type: str, shape: Tuple[int, ...]) -> str:
    num_dims = len(shape)
    shape_str = ", ".join([str(d) for d in shape[1:]])
    base = f"Tensor{num_dims}D<{data_type}, {shape_str}>"
    return f"""
    class {class_name} : public spio::{base} {{
    public:
        using Base = {base};

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
    return f"""
    }};
"""
