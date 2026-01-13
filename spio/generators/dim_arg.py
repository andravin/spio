"""Type alias and utilities for dimension arguments."""

from typing import Union

from .dim import Dim
from .fold import Fold


# Type alias for dimension arguments that can be str, Dim, or Fold
DimArg = Union[str, Dim, Fold]


def normalize_dim_arg(arg: DimArg) -> str:
    """Normalize a dimension argument to a string dimension name.

    Accepts:
        - str: returned as-is (uppercased)
        - Dim: returns dim.dim_name
        - Fold: returns fold.fold_name

    Raises:
        ValueError: If the Dim/Fold doesn't have a name set.
    """
    if isinstance(arg, str):
        return arg.upper()

    if isinstance(arg, Dim):
        if arg.dim_name is None:
            raise ValueError(
                "Dim must have a name set before use. "
                "Assign to a Generators attribute or create with Dim('name')."
            )
        return arg.dim_name

    if isinstance(arg, Fold):
        if arg.fold_name is None:
            raise ValueError(
                "Fold must have a name set before use. "
                "Assign to a Generators attribute or create with an explicit name."
            )
        return arg.fold_name

    raise TypeError(f"Expected str, Dim, or Fold, got {type(arg).__name__}")
