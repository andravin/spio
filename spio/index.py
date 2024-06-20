from typing import Dict, Tuple


def index_header():
    """Return the C++ statement that includes the spio index header file.
    
    This file implements the C++ base template classes from which the
    custom index classes inherit. You must include this header before using
    the code returned by the generate_index() function.
    """
    return '#include "spio/index.h"'


def generate_index(class_name: str, dims: Dict[str, int]) -> str:
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
    code += _tail()
    return code


def _class(class_name: str, shape: Tuple[int, ...]) -> str:
    num_dims = len(shape)
    shape_str = ", ".join([str(d) for d in shape[1:]])
    base = f"Index{num_dims}D<{shape_str}>"
    return f"""
    class {class_name} : public {base} {{
    public:
        using Base = {base};

        constexpr {class_name}(int offset  = 0) : Base(offset) {{}}

        constexpr {class_name}(const {base} &other) : Base(other) {{}}
"""


def _dim(name: str, value: int) -> str:
    name = name.upper()
    return f"""
        constexpr static int {name} = {value};
    """


def _index_to_offset(class_name: str, d: int, name: str) -> str:
    name_in = f"{name}_in"
    dim_d = f"_d{d}"
    return f"""
        constexpr {class_name} {name}(int {name_in}) const {{ return {dim_d}({name_in}); }}
"""


def _offset_to_index(d: int, name: str) -> str:
    dim_d = f"_d{d}"
    return f"""
        constexpr int {name}() const {{ return {dim_d}(); }}
"""


def _tail() -> str:
    return f"""
    }};
"""
