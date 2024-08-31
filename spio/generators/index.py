from math import prod
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class IndexSpec:
    class_name: str
    dims: Dict[str, int]

    def generate(self) -> str:
        return _generate_index(self.class_name, self.dims)


def generate_indices(index_specs: List[IndexSpec], header_file_name: str) -> None:
    code = _index_header()
    code += ""
    for index_spec in index_specs:
        code += index_spec.generate()
        code += "\n"

    with open(header_file_name, "w") as file:
        file.write(code)


def _generate_index(class_name: str, dims: Dict[str, int]) -> str:
    """Return the C++ source code that implements a custom index class.

    Custom index classes use named tensor dimensions.

    Parameters:
        class_name(str): the name to use for the C++ class
        dims(Dict[str, int]): an (ordered) dict that maps dimension names to their sizes.
    """
    code = ""
    code += _class(class_name, tuple(dims.values()))
    for name, value in dims.items():
        code += _dim(name, value)
    for d, name in enumerate(dims.keys()):
        code += _index_to_offset(class_name, d, name)
    for d, name in enumerate(dims.keys()):
        code += _offset_to_index(d, name)
    code += _size(dims.values())
    code += _tail()
    return code


def _index_header():
    """Return the C++ statement that includes the spio index header file.

    This file implements the C++ base template classes from which the
    custom index classes inherit. You must include this header before using
    the code returned by the generate_index() function.
    """
    return '#include "spio/index.h"'


def _class(class_name: str, shape: Tuple[int, ...]) -> str:
    num_dims = len(shape)
    shape_str = ", ".join([str(d) for d in shape[1:]])
    base = f"Index{num_dims}D<{shape_str}>"
    return f"""
    class {class_name} : public spio::{base} {{
    public:
        using Base = {base};

        DEVICE constexpr {class_name}(unsigned offset  = 0) : Base(offset) {{}}

        DEVICE constexpr {class_name}(const {base} &other) : Base(other) {{}}
"""


def _dim(name: str, value: int) -> str:
    name = name.upper()
    return f"""
        static constexpr unsigned {name} = {value};
    """


def _size(dims: List[int]) -> str:
    size = prod(dims)
    return f"""
        static constexpr unsigned size = {size};
"""


def _index_to_offset(class_name: str, d: int, name: str) -> str:
    name_in = f"{name}_in"
    dim_d = f"_d{d}"
    return f"""
        DEVICE constexpr {class_name} {name}(unsigned {name_in}) const {{ return {dim_d}({name_in}); }}
"""


def _offset_to_index(d: int, name: str) -> str:
    dim_d = f"_d{d}"
    return f"""
        DEVICE constexpr unsigned {name}() const {{ return {dim_d}(); }}
"""


def _tail() -> str:
    return f"""
    }};
"""
