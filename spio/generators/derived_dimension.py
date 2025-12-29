"""Module for derived dimension generators."""

from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class DerivedDimension(Protocol):
    """Protocol for derived dimension generators."""

    def get_class_name(self) -> str:
        """Return the class name of the derived dimension."""


@runtime_checkable
class SizedDerivedDimension(DerivedDimension, Protocol):
    """Protocol for single-output derived dimensions.

    These can be used inline in Dims() because they have a definite size
    and can be bound to a dimension name.
    """

    def set_output_dim_name(self, name: str) -> None:
        """Set the name of the output dimension."""

    @property
    def size(self) -> int:
        """Return the size of the derived dimension."""
