"""Module for derived dimension generators."""

from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class DerivedDimension(Protocol):
    """Protocol for derived dimension generators."""

    @property
    def size(self) -> int:
        """Return the size of the derived dimension."""


@runtime_checkable
class SupportsOutputDimName(Protocol):
    """Protocol for derived dimensions that support setting output dim name.

    This only makes sense for DeriveDimension subclasses that have a single output dimension.
    """

    def set_output_dim_name(self, name: str) -> None:
        """Set the name of the output dimension."""
