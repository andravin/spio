"""Type alias and utilities for dimension arguments."""

from typing import Union

from .dim import Dim, StaticDim
from .fold import Fold, StaticFold


# Type alias for dimension arguments that can be str, Dim, Fold, StaticDim, or StaticFold
DimArg = Union[str, Dim, Fold, StaticDim, StaticFold]


def extract_dim_from_arg(arg: DimArg) -> Union[Dim, Fold]:
    """Extract the underlying Dim or Fold object from a dimension argument.

    Accepts:
        - Dim: returned as-is
        - Fold: returned as-is
        - StaticDim: returns the underlying Dim
        - StaticFold: returns the underlying Fold
        - str: raises TypeError (cannot extract Dim from string)

    Returns:
        The underlying Dim or Fold object.

    Raises:
        TypeError: If arg is a string (use normalize_dim_arg for string handling).
    """
    if isinstance(arg, StaticDim):
        return arg.dim
    if isinstance(arg, StaticFold):
        return arg.fold
    if isinstance(arg, (Dim, Fold)):
        return arg
    if isinstance(arg, str):
        raise TypeError(
            f"Cannot extract Dim/Fold from string '{arg}'. "
            f"Pass a Dim, Fold, StaticDim, or StaticFold object."
        )
    raise TypeError(
        f"Expected Dim, Fold, StaticDim, or StaticFold, got {type(arg).__name__}"
    )


def normalize_dim_arg(arg: DimArg) -> str:
    """Normalize a dimension argument to a string dimension name.

    Accepts:
        - str: returned as-is (uppercased)
        - Dim: returns dim.dim_name
        - Fold: returns fold.fold_name
        - StaticDim: returns the underlying dim.dim_name
        - StaticFold: returns the underlying fold.fold_name

    Raises:
        ValueError: If the Dim/Fold doesn't have a name set.
    """
    if isinstance(arg, str):
        return arg.upper()

    # Handle StaticDim/StaticFold by extracting the underlying Dim/Fold
    if isinstance(arg, StaticDim):
        arg = arg.dim
    elif isinstance(arg, StaticFold):
        arg = arg.fold

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

    raise TypeError(
        f"Expected str, Dim, Fold, StaticDim, or StaticFold, got {type(arg).__name__}"
    )
