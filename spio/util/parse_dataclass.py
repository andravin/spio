"""Functions for parsing dataclasses from strings."""

import ast
import re
from dataclasses import dataclass
from typing import Dict, Type, Any


def parse_dataclass(expr: str, dataclasses: Dict[str, Type[dataclass]] = None) -> Any:
    """Parse a dataclass instance from a string expression."""
    expr = expr.strip()
    if expr:
        try:
            # Extract the dataclass name and parameters
            match = re.match(r"(\w+)\((.*)\)", expr)
            if not match:
                raise ValueError(f"Invalid expression format: '{expr}'")

            dataclass_name, params_str = match.groups()

            # Convert parameters string to dictionary format
            params_str = re.sub(r"(\w+)=", r'"\1":', params_str)
            data = ast.literal_eval(f"{{{params_str}}}")

            if not isinstance(data, dict):
                raise ValueError(
                    f"Expression must evaluate to a dictionary, got {type(data).__name__}"
                )

            if dataclasses and dataclass_name in dataclasses:
                dc = dataclasses[dataclass_name]
                return dc(**data)
            raise ValueError(
                f"No matching dataclass found for the name '{dataclass_name}'"
            )
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse line '{expr}'") from e
    return None
