from dataclasses import dataclass


def parse_dataclass(expr: str, dataclasses=None) -> dataclass:
    expr = expr.strip()
    if expr:
        try:
            return eval(expr, dataclasses)
        except (SyntaxError, NameError) as e:
            raise ValueError(f"Failed to parse line '{expr}': {e}")
    return None
