"""Code generator for custom folded-dimension template classes."""

from dataclasses import dataclass
from typing import Tuple

from .gen_specs import GenSpecs
from .dim import (
    dim_name_to_dim_or_fold_class_name,
    _format_fold_template_instance,
    _format_dim_class_name,
)


@dataclass
class FoldSpec(GenSpecs):
    """CUDA Code generator for custom folded-dimension classes.

    This class defines a folding of a tensor dimension. The
    tensor dimension must already have been generated using DimSpec.
    """

    fold_name: str
    dim_name: str
    stride: int

    def generate(self) -> str:
        dim_class_name = dim_name_to_dim_or_fold_class_name(self.dim_name)
        fold_template_instance = _format_fold_template_instance(
            dim_class_name, self.stride
        )
        fold_class_name = _format_dim_class_name(self.fold_name)
        return f"""
using {fold_class_name} = {fold_template_instance};
"""

    @property
    def dim_names(self) -> Tuple[str]:
        """Return the name of the folded dimension."""
        return (self.dim_name,)


def dim_header() -> str:
    """Return a C++ statement that includes the spio dim header.

    The header implements the C++ spio::Fold template class.
    """
    return '#include "spio/fold.h"'
