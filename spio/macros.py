from typing import Dict
from dataclasses import dataclass


@dataclass
class MacroSpec:
    macros: Dict[str, str]

    def generate(self) -> str:
        code = ""
        for name, value in self.macros.items():
            code += f"#define {name} {value}\n"
        return code
