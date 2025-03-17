"""Protocol for kernel code generation classes."""

from typing import Protocol


class GenSpecs(Protocol):
    """Protocol for kernel code generation classes.

    Used by code generators for named tensors, constant variables,
    macros, and other kernel-specific structures that are used in the
    CUDA kernel source code.

    See classes in spio.generators for examples.
    """

    def generate(self) -> str:
        """Generate CUDA source code."""


class GenSpecsWithContext(GenSpecs, Protocol):
    """Protocol for kernel code generation classes with optional context."""

    def generate_with_context(self, user_data_types: list[str] = None) -> str:
        """Generate CUDA source code with context.

        @param user_data_types: List of user-defined types.
        User-defined types currently include Fragment class-names. Tensors
        allow user-defined types to be used as the tensors data-type.
        """

    def generate(self) -> str:
        """Generate CUDA source code without context.

        This method is provided for backwards compatibility with the GenSpecs protocol.
        """
        return self.generate_with_context(user_data_types=None)
