"""Code generator for custom dimension classes in CUDA / C++."""

from dataclasses import dataclass
from typing import Tuple

from .gen_specs import GenSpecs


@dataclass
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

    def generate(self) -> str:
        class_name = self.class_name
        return f"""
class {class_name} : public spio::Dim
{{
public:
    using Base = spio::Dim;
    using Base::Base;
    class Iterator {{
    public:
        DEVICE constexpr Iterator(int i) : _i(i) {{}}
        DEVICE constexpr {class_name} operator*() const {{ return _i; }}
        DEVICE constexpr Iterator &operator++() {{ ++_i; return *this; }}
        DEVICE constexpr bool operator!=(const Iterator other) const {{ return _i != other._i; }}

    private:
        int _i;
    }};
    template <class NewDimType>
    DEVICE constexpr auto cast() const -> NewDimType {{ return NewDimType(Base::get()); }}
    template <unsigned Stride>
    DEVICE constexpr spio::Fold<{class_name}, Stride> fold() const {{ return spio::Fold<{class_name}, Stride>(*this); }}
    DEVICE constexpr Iterator begin() const {{ return Iterator(0); }}
    DEVICE constexpr Iterator end() const {{ return Iterator(Base::get()); }}
    DEVICE constexpr bool operator<(const {class_name} other) const {{ return Base::operator<(other); }}
    DEVICE constexpr bool operator>(const {class_name} other) const {{ return other < *this; }}
    DEVICE constexpr bool operator<=(const {class_name} other) const {{ return !(*this > other); }}
    DEVICE constexpr bool operator>=(const {class_name} other) const {{ return !(*this < other); }}
    DEVICE constexpr bool operator==(const {class_name} other) const {{ return Base::operator==(other); }}
    DEVICE constexpr bool operator!=(const {class_name} other) const {{ return !(*this == other); }}
    DEVICE constexpr {class_name} operator+(const {class_name} other) const {{ return {class_name}(Base::_add(other)); }}
    DEVICE constexpr {class_name} operator-(const {class_name} other) const {{ return {class_name}(Base::_sub(other)); }}
    DEVICE constexpr {class_name} operator%(const {class_name} other) const {{ return {class_name}(Base::_modulus(other)); }}
}};
"""

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


def _get_dim_name_and_stride(name: str) -> str:
    """Convert a dimension name to a dimension class name."""
    stride = None
    for i, char in enumerate(name):
        if char.isdigit():
            stride = int(name[i:])
            name = name[:i]
            break
    return name, stride


def _format_dim_class_name(dim_name: str) -> str:
    """Convert a dimension name to a dimension class name."""
    return f"{dim_name.upper()}_Dim"


def _format_fold_template_instance(dim_class_name: str, stride: int) -> str:
    return f"spio::Fold<{dim_class_name}, {stride}>"


def header() -> str:
    """Return a C++ statement that includes the spio dim header.

    The header implements the C++ spio::Dim base class from which the
    custom dimension classes inherit.
    """
    return '#include "spio/dim.h"'
