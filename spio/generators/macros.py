"""Code generator for macros in CUDA kernel source code."""
from typing import Dict
from dataclasses import dataclass


@dataclass
class Macro:
    """Code generator for macros in CUDA kernel source code.

    This class is used to generate macro definitions for the CUDA kernel source code.

    Attributes:
        macros (Dict[str, str]): A dictionary of macro names and their corresponding values.
    """

    macros: Dict[str, str]

    def generate(self) -> str:
        """Generate the macro definitions as a string."""
        code = ""
        for name, value in self.macros.items():
            code += f"#define {name} {value}\n"
        return code
