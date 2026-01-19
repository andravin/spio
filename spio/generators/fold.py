"""Code generator for custom folded-dimension template classes."""

from typing import Tuple, Union
from dataclasses import dataclass

from .gen_specs import GenSpecs
from .dim import (
    Dim,
    dim_name_to_dim_or_fold_class_name,
    _format_fold_template_instance,
    _format_dim_class_name,
    _get_dim_name_and_stride,
)
from .built_in import BuiltIn


@dataclass(frozen=True, eq=False)
class StaticFold:
    """A folded dimension with a statically known size.

    Created by calling a Fold object with a size argument: K8(4).

    Attributes:
        fold: The Fold object this sized fold is based on.
        size: The size of the folded dimension.
    """

    fold: "Fold"
    size: int

    @property
    def dim_name(self) -> str:
        """Return the fold name (used as the dimension name in Dims)."""
        return self.fold.fold_name

    def _identity_key(self) -> tuple:
        """Return a key for object-based matching.

        Two StaticFolds match if they have the same base Dim (by identity)
        and the same stride. This allows J.fold(8) to match even when
        called in different places.
        """
        return (id(self.fold.dim), self.fold.stride)

    def __eq__(self, other) -> bool:
        if not isinstance(other, StaticFold):
            return NotImplemented
        return self._identity_key() == other._identity_key()

    def __hash__(self) -> int:
        return hash((type(self), self._identity_key()))

    def __mul__(self, other: "StaticFold") -> "StaticFold":
        """Multiply two StaticFolds with the same base Dim and stride.

        Returns a new StaticFold with sizes added together.
        """
        if isinstance(other, int):
            return StaticFold(fold=self.fold, size=self.size * other)
        if not isinstance(other, StaticFold):
            return NotImplemented
        if self._identity_key() != other._identity_key():
            raise TypeError(
                f"Cannot multiply StaticFolds with different base Dim or stride: "
                f"{self.fold.dim_name}/{self.fold.stride} vs "
                f"{other.fold.dim_name}/{other.fold.stride}"
            )
        return StaticFold(fold=self.fold, size=self.size + other.size)

    def __rmul__(self, other: "StaticFold") -> "StaticFold":
        """Right multiply - handles int * StaticFold."""
        if isinstance(other, int):
            return StaticFold(fold=self.fold, size=self.size * other)
        return self.__mul__(other)

    def __truediv__(self, divisor: int) -> "StaticFold":
        """Create a coarser StaticFold by increasing the stride by the given divisor.

        The new stride is self.fold.stride * divisor, and the new size is
        self.size // divisor.

        Example:
            k8 = K8(16)  # stride 8, size 16
            k16 = k8 / 2  # stride 16, size 8

        Raises:
            ValueError: If divisor does not evenly divide self.size.
        """
        if not isinstance(divisor, int):
            return NotImplemented
        if self.size % divisor != 0:
            raise ValueError(
                f"Divisor {divisor} is not divisible into size {self.size}."
            )
        new_stride = self.fold.stride * divisor
        new_fold = self.fold.fold(new_stride)
        new_size = self.size // divisor
        return StaticFold(fold=new_fold, size=new_size)

    def __mod__(self, modulus: int) -> "StaticFold":
        """Create a StaticFold with size set to the modulus.

        Example:
            k8 = K8(32)  # stride 8, size 18
            k8_mod = k8 % 4  # stride 8, size 4
        """
        return StaticFold(fold=self.fold, size=modulus)


@dataclass(frozen=True)
class Fold(GenSpecs):
    """CUDA Code generator for custom folded-dimension classes.

    This class defines a folding of a tensor dimension. The
    tensor dimension must already have been generated using DimSpec.

    When used with the Generators container, fold_name can be omitted and will
    be set from the attribute name.

    Can be created in two ways:
    - Legacy: Fold("K", 8) or Fold(dim_name="K", stride=8)
    - New: K.fold(8) where K is a Dim object

    Attributes:
        dim: Either a Dim object or a string name of the base dimension to fold.
        stride: The fold stride.
        fold_name: The name of the folded dimension class (optional with Generators).
        init: Optional built-in initializer.

    Example:
        # Using Dim.fold() method (preferred)
        K = Dim()
        K8 = K.fold(8)
        Dims(K(32), K8(4))

        # Legacy string-based
        K8 = Fold("K", 8)
    """

    dim: Union[str, Dim]
    stride: int
    fold_name: str = None
    init: BuiltIn = None

    def __post_init__(self):
        """Normalize the fold and dimension names to upper-case."""
        if self.fold_name is not None:
            object.__setattr__(self, "fold_name", self.fold_name.upper())
        # If dim is a string, normalize to upper-case
        if isinstance(self.dim, str):
            object.__setattr__(self, "dim", self.dim.upper())

    def __call__(self, size: int) -> StaticFold:
        """Create a StaticFold with this fold and the given size.

        Args:
            size: The size of the folded dimension.

        Returns:
            A StaticFold object representing this fold with the given size.

        Example:
            K = Dim()
            K8 = K.fold(8)
            K8(4)  # StaticFold with fold=K8 and size=4
        """
        return StaticFold(fold=self, size=size)

    def fold(self, new_stride: int) -> "Fold":
        """Create a fold with a new stride."""
        base_dim = self.dim if isinstance(self.dim, Dim) else None
        if base_dim is not None:
            return base_dim.fold(new_stride)
        # Legacy string-based fold
        return Fold(dim=self.dim, stride=new_stride)

    def __truediv__(self, divisor: int) -> "Fold":
        """Create a coarser Fold by increasing the stride by the divisor.

        Equivalent to self.fold(self.stride * divisor).
        """
        if not isinstance(divisor, int):
            return NotImplemented
        return self.fold(self.stride * divisor)

    @property
    def dim_name(self) -> str:
        """Return the base dimension name as a string."""
        if isinstance(self.dim, str):
            return self.dim
        return self.dim.dim_name

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
