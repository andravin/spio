"""Code generator for custom index classes in CUDA kernel source
code."""

from math import prod
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class IndexSpec:
    """CUDA Code generator for custom index classes.

    This class is used to generate custom index classes that map named tensor dimensions to offsets.
    Conversely, it can also map offsets back to named tensor dimensions.

    Attributes:
        class_name (str): The name of the custom index class.
        dims (Dict[str, int]): A dictionary mapping dimension names to their sizes.
    """

    class_name: str
    dims: Dict[str, int]

    def generate(self) -> str:
        """Generate the C++ source code for the custom index class."""
        return _generate_index(self.class_name, self.dims)


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
        code += _offset_to_index(d, name)
    code += _size(dims.values())
    code += _tail()
    return code


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


def _offset_to_index(d: int, name: str) -> str:
    dim_d = f"_d{d}"
    return f"""
        DEVICE constexpr int {name}() const {{ return {dim_d}(); }}
"""


def _tail() -> str:
    return """
    };
"""


def index_header() -> str:
    """Return a C++ statement that includes the spio index header.

    The header implements the C++ base template classes from which the
    custom index classes inherit.
    """
    return '#include "spio/index.h"'
