"""Code generator for macros in CUDA kernel source code."""

from typing import Dict
from dataclasses import dataclass

from .gen_specs import GenSpecs


@dataclass
class Macro(GenSpecs):
    """Code generator for macros in CUDA kernel source code.

    This class is used to generate macro definitions for the CUDA kernel source code.

    Attributes:
        macros (Dict[str, str]): A dictionary of macro names and their corresponding values.
    """

    macros: Dict[str, str]

    def _set_class_name(self, name: str) -> None:
        """Set the class name (ignored for Macro).

        Macro doesn't use a class name since it generates #define statements.
        This method exists for compatibility with the Generators container.
        """
        pass

    def get_class_name(self) -> str:
        """Return None - Macro doesn't have a class name."""
        return None

    def generate(self) -> str:
        """Generate the macro definitions as a string."""
        code = ""
        for name, value in self.macros.items():
            code += f"#define {name} {value}\n"
        return code
