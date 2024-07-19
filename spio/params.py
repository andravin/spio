from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ParamsSpec:
    name_space: str
    params: Dict[str, Any]

    def generate(self) -> None:
        code = f"namespace {self.name_space} {{\n"
        for name, val in self.params.items():
            c_type_name = _c_type_name(val)
            code += f"    inline constexpr {c_type_name} {name} = {val};\n"
        code += "}\n"
        return code


def _c_type_name(val: Any) -> str:
    if isinstance(val, int):
        return "int"
    elif isinstance(val, float):
        return "float"
    else:
        raise ValueError(f"Unsupported parameter type {type(val)}")
