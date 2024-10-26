"""Functions for parsing dataclasses from strings."""

from dataclasses import dataclass
from typing import List, Type


def parse_dataclass(expr: str, dataclasses: List[Type[dataclass]] = None) -> dataclass:
    """Parse a dataclass instance from a string expression."""
    expr = expr.strip()
    if expr:
        try:
            # pylint: disable=eval-used
            return eval(expr, dataclasses)
        except (SyntaxError, NameError) as e:
            raise ValueError(f"Failed to parse line '{expr}'") from e
    return None
