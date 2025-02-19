"""Code generator for custom index classes in CUDA / C++."""

from math import prod
from typing import Dict, Tuple, List, Union
from dataclasses import dataclass

from .gen_specs import GenSpecs
from .subindex_protocol import SubindexProtocol


@dataclass
class IndexSpec(GenSpecs):
    """CUDA Code generator for custom index classes.

    This class is used to generate custom index classes that map named tensor dimensions to offsets.
    Conversely, it can also map offsets back to named tensor dimensions.

    Attributes:
        class_name (str): The name of the custom index class.
        dims (Dict[str, int]): A dictionary mapping dimension names to their sizes.
    """

    class_name: str
    dims: Dict[str, Union[int, SubindexProtocol]]

    def generate(self) -> str:
        """Generate the C++ source code for the custom index class."""
        return _generate_index(self.class_name, self.dims)

    @property
    def size(self) -> int:
        """Return the total number of elements in the index."""
        return _calc_size(_sizes_gen(self.dims))


def index_header() -> str:
    """Return a C++ statement that includes the spio index header.

    The header implements the C++ base template classes from which the
    custom index classes inherit.
    """
    return '#include "spio/index.h"'


def _generate_index(class_name: str, dims: Dict[str, Union[int, SubindexProtocol]]) -> str:
    """Return the C++ source code that implements a custom index class.

    Custom index classes use named tensor dimensions.

    Parameters:
        class_name(str): the name to use for the C++ class
        dims(Dict[str, int]): an (ordered) dict that maps dimension names to their sizes.
    """
    code = ""
    sizes = _sizes_gen(dims)
    code += _class(class_name, sizes)
    for name, value in dims.items():
        if isinstance(value, SubindexProtocol):
            value.class_name = _fused_dim_class_name(name)
            code += value.generate()
    for name, size in zip(dims.keys(), sizes):
        code += _dim(name, size)
    for d, (name, value) in enumerate(dims.items()):
        code += _offset_to_index(d, name, value)
    code += _size(_calc_size(sizes))
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


def _fused_dim_class_name(fused_dim_name: str) -> str:
    return f"_{fused_dim_name.capitalize()}Idx"


def _dim(name: str, size: int) -> str:
    name = name.upper()
    return f"""
        static constexpr unsigned {name} = {size};
    """


def _size(size) -> str:
    return f"""
        static constexpr unsigned size = {size};
"""


def _offset_to_index(d: int, name: str, value: Union[int, SubindexProtocol]) -> str:
    if isinstance(value, SubindexProtocol):
        return _offset_to_fused_idx(d, name, value)
    else:
        return _offset_to_int_index(d, name)


def _offset_to_int_index(d: int, name: str) -> str:
    dim_d = f"_d{d}"
    return f"""
        DEVICE constexpr int {name}() const {{ return {dim_d}(); }}
"""


def _offset_to_fused_idx(d: int, name: str, fused_dim_spec: SubindexProtocol) -> str:
    dim_d = f"_d{d}"
    return f"""
        DEVICE constexpr {fused_dim_spec.class_name} {name}() const {{
            const auto offset = {dim_d}();
            return {fused_dim_spec.class_name}(offset);
        }}
    """


def _tail() -> str:
    return """
    };
"""


def _sizes_gen(dims: Dict[str, int]):
    return [d.size if isinstance(d, SubindexProtocol) else d for d in dims.values()]


def _calc_size(sizes: List[int]):
    return prod(sizes)
