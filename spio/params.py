from typing import Dict, Any


def generate_params(
    name_space: str, params: Dict[str, Any], header_file_name: str
) -> None:
    """Write a C++/CUDA header file that defines the given constant parameters.

    Arguments:
        name_space: The C++ namespace in which the constants will be placed.
        params: A dictionary of parameter names and values.
        header_file_name: The header file to write.
    """
    code = f"namespace {name_space} {{\n"
    for name, val in params.items():
        c_type_name = _c_type_name(val)
        code += f"    inline constexpr {c_type_name} {name} = {val};\n"
    code += "}\n"

    with open(header_file_name, "w") as file:
        file.write(code)


def _c_type_name(val: Any) -> str:
    if isinstance(val, int):
        return "int"
    elif isinstance(val, float):
        return "float"
    else:
        raise ValueError(f"Unsupported parameter type {type(val)}")
