"""Code generator for custom dimension classes in CUDA / C++."""

from dataclasses import dataclass
from typing import Tuple

from .gen_specs import GenSpecs


@dataclass(frozen=True)
class Dim(GenSpecs):
    """CUDA Code generator for custom dimension classes.

    This class defines a named tensor dimension.

    Note: the spio.generators.generate() method will automatically detect
    all dimensions used in the generator specifications and generate the
    corresponding custom dimension classes. Normally the user will not
    need to use this class directly.

    Attributes:
        class_name (str): The name of the custom dimension class.
    """

    dim_name: str

    @property
    def class_name(self) -> str:
        """Convert the dimension name to a dimension class name."""
        return _format_dim_class_name(self.dim_name)

    def generate(self):
        """Generate the C++ code for a dimension class using CRTP."""
        class_name = self.class_name
        return f"class {class_name} : public spio::Dim<{class_name}> {{ public: using spio::Dim<{class_name}>::Dim; }};\n"

    @property
    def dim_names(self) -> Tuple[str,]:
        """Return the name of the dimension."""
        return tuple(
            self.dim_name,
        )


def dim_name_to_dim_or_fold_class_name(name: str) -> str:
    """Convert a dimension name to a dimension or folded-dimension class name."""
    dim_class_name, dim_stride = _get_dim_name_and_stride(name)
    return _get_dim_or_fold_class_name(dim_class_name, dim_stride)


def _get_dim_or_fold_class_name(name: str, stride: int):
    dim_class_name = _format_dim_class_name(name)
    if stride is None:
        return dim_class_name
    else:
        return _format_fold_template_instance(dim_class_name, stride)


def _get_dim_name_and_stride(name: str) -> Tuple[str, int]:
    """Convert a dimension name to a dimension name and stride."""
    stride = None
    for i, char in enumerate(name):
        if char.isdigit():
            stride = int(name[i:])
            name = name[:i]
            break
    return name, stride


def _format_dim_class_name(dim_name: str) -> str:
    """Convert a dimension name to a dimension class name."""
    return f"{dim_name.upper()}"


def _format_fold_template_instance(dim_class_name: str, stride: int) -> str:
    return f"spio::Fold<{dim_class_name}, {stride}>"


def header() -> str:
    """Return a C++ statement that includes the spio dim header.

    The header implements the C++ spio::Dim base class from which the
    custom dimension classes inherit.
    """
    return '#include "spio/dim.h"'
