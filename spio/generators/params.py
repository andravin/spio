"""Code generator for constant parameters in CUDA kernel source code."""

from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ParamsSpec:
    """CUDA Code generator for constant parameters.

    The parameters are defined in a C++ namespace. Parameters may be of type
    bool, int, or float.

    Attributes:
        name_space: The name of the C++ namespace.
        params: A dictionary of parameter names and their values.
    """

    name_space: str
    params: Dict[str, Any]

    def generate(self) -> str:
        """Generate the C++ code for the parameter definitions."""
        code = f"namespace {self.name_space} {{\n"
        for name, val in self.params.items():
            c_type_name, c_value = _c_type_name(val)
            code += f"    inline constexpr {c_type_name} {name} = {c_value};\n"
        code += "}\n"
        return code


def _c_type_name(val: Any) -> Tuple[str, Any]:
    if isinstance(val, bool):
        # Careful, bool is int, so check for bool first.
        return "bool", "true" if val else "false"
    if isinstance(val, int):
        return "int", val
    if isinstance(val, float):
        return "float", val
    raise ValueError(f"Unsupported parameter type {type(val)}")
