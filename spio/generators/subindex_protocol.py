"""Protocol for index code-generation classes."""

from typing import Protocol, runtime_checkable

from .gen_specs import GenSpecs


@runtime_checkable
class SubindexProtocol(GenSpecs, Protocol):
    """Protocol for index classes that are meant to be embedded inside other index classes.

    Subindex classes can be used for any of the dimensions in IndexSpec and TensorSpec definitions.
    See CheckerboardIndexSpec for an example.
    """

    @property
    def size(self) -> int:
        """Return the compound size of the index dimensions."""

    def generate_offset_function_declaration(
        self, return_type: str, function_name: str
    ) -> str:
        """Return the CUDA / C++ source code for the index offset function declaration.

        The offset function returns the offset as a function of the index dimensions.
        """

    def generate_offset_function_call(self) -> str:
        """Return the CUDA / C++ source code for the offset function call."""
