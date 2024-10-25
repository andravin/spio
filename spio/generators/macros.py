from typing import Dict
from dataclasses import dataclass


@dataclass
class MacroSpec:
    """Code generator for macros in CUDA kernel source code.

    This class is used to generate macro definitions for the CUDA kernel source code.

    Attributes:
        macros (Dict[str, str]): A dictionary of macro names and their corresponding values.
    """

    macros: Dict[str, str]

    def generate(self) -> str:
        code = ""
        for name, value in self.macros.items():
            code += f"#define {name} {value}\n"
        return code
