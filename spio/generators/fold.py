"""Code generator for custom folded-dimension template classes."""

from typing import Tuple
from dataclasses import dataclass

from .gen_specs import GenSpecs
from .dim import (
    dim_name_to_dim_or_fold_class_name,
    _format_fold_template_instance,
    _format_dim_class_name,
    _get_dim_name_and_stride,
)
from .built_in import BuiltIn


@dataclass(frozen=True)
class Fold(GenSpecs):
    """CUDA Code generator for custom folded-dimension classes.

    This class defines a folding of a tensor dimension. The
    tensor dimension must already have been generated using DimSpec.

    When used with the Generators container, fold_name can be omitted and will
    be set from the attribute name.

    Attributes:
        dim_name: The name of the base dimension to fold.
        stride: The fold stride.
        fold_name: The name of the folded dimension class (optional with Generators).
    """

    dim_name: str
    stride: int
    fold_name: str = None
    init: BuiltIn = None

    def __post_init__(self):
        """Normalize the fold and dimension names to upper-case."""
        if self.fold_name is not None:
            object.__setattr__(self, "fold_name", self.fold_name.upper())
        object.__setattr__(self, "dim_name", self.dim_name.upper())

    def _set_class_name(self, name: str) -> None:
        """Set the fold name for this fold.

        Called by the Generators container when the fold is assigned to an attribute.
        """
        object.__setattr__(self, "fold_name", name.upper())

    def get_class_name(self) -> str:
        """Return the fold name, or None if not set."""
        return self.fold_name

    def generate(self):
        dim_class_name = dim_name_to_dim_or_fold_class_name(self.dim_name)
        fold_template_instance = _format_fold_template_instance(
            dim_class_name, self.stride
        )
        fold_class_name = _format_dim_class_name(self.fold_name)

        if self.init is None:
            return f"using {fold_class_name} = {fold_template_instance};\n"

        # Generate a derived struct with an initializing constructor
        return (
            f"struct {fold_class_name} : {fold_template_instance} {{\n"
            f"    {fold_class_name}() : {fold_template_instance}({self.init.value}) {{}}\n"
            f"}};\n"
        )

    @property
    def dim_names(self) -> Tuple[str]:
        """Return the base dimension name, not the folded form.

        This ensures we don't create redundant dimension classes for already folded dimensions.
        """
        base_name, _ = _get_dim_name_and_stride(self.dim_name)
        return (base_name,)


def dim_header() -> str:
    """Return a C++ statement that includes the spio dim header.

    The header implements the C++ spio::Fold template class.
    """
    return '#include "spio/fold.h"'
