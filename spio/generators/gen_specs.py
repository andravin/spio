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
