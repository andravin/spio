"""Protocol for kernel code generation classes."""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class GenSpecs(Protocol):
    """Protocol for kernel code generation classes.

    Used by code generators for named tensors, constant variables,
    macros, and other kernel-specific structures that are used in the
    CUDA kernel source code.

    See classes in spio.generators for examples.
    """

    def generate(self) -> str:
        """Generate CUDA source code."""

    def used_generators(self) -> list["GenSpecs"]:
        """Return a list of generator objects used by this generator.

        This is used to find any generators that need to be assigned class names automatically.
        Subclasses should override this method if they use other generators.
        """
        return []

    def get_class_name(self) -> Optional[str]:
        """Return the class/type name this generator produces, or None if not yet set.

        Used to detect unnamed generators that need auto-naming.
        Each subclass should implement this based on its naming convention
        (e.g., class_name, fold_name, coord_name, etc.).
        """
        return None


@runtime_checkable
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
