"""Module defining a protocol for layer parameters."""

from typing import Protocol


class Params(Protocol):
    """Protocol for layer parameters."""

    def encode(self) -> str:
        """Encode parameters to a string representation."""

    @classmethod
    def decode(cls, string: str) -> "Params":
        """Decode parameters from a string representation."""

    def is_valid(self) -> bool:
        """Validate the parameters.

        Returns True if the parameters are valid, False otherwise.
        """

    def validate(self) -> None:
        """Validate the parameters and raise an exception if invalid."""
