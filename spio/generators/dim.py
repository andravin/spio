"""Code generator for custom dimension classes in CUDA / C++."""

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

from .gen_specs import GenSpecs
from ..util import divup

if TYPE_CHECKING:
    from .fold import Fold, StaticFold


@dataclass(frozen=True, eq=False)
class StaticDim:
    """A dimension with a statically known size.

    Created by calling a Dim object with a size argument: I(16).

    Attributes:
        dim: The Dim object this sized dimension is based on.
        size: The size of the dimension.
    """

    dim: "Dim"
    size: int

    @property
    def dim_name(self) -> str:
        """Return the dimension name."""
        return self.dim.dim_name

    def _identity_key(self) -> int:
        """Return a key for object-based matching.

        Two StaticDims match if they have the same base Dim (by identity).
        """
        return id(self.dim)

    def __eq__(self, other) -> bool:
        if not isinstance(other, StaticDim):
            return NotImplemented
        return self._identity_key() == other._identity_key()

    def __hash__(self) -> int:
        return hash(self._identity_key())

    def __mul__(self, other: "StaticDim") -> "StaticDim":
        """Multiply two StaticDims with the same base Dim."""
        if isinstance(other, int):
            return StaticDim(dim=self.dim, size=self.size * other)
        if not isinstance(other, StaticDim):
            return NotImplemented
        if self.dim is not other.dim:
            raise TypeError(
                f"Cannot multiply StaticDims with different base Dims: "
                f"{self.dim.dim_name} vs {other.dim.dim_name}"
            )
        return StaticDim(dim=self.dim, size=self.size + other.size)

    def __rmul__(self, other: "StaticDim") -> "StaticDim":
        """Right multiply - handles int * StaticDim."""
        if isinstance(other, int):
            return StaticDim(dim=self.dim, size=self.size * other)
        return self.__mul__(other)

    def __add__(self, other: "StaticDim") -> "StaticDim":
        """Add two StaticDims with the same base Dim."""
        if isinstance(other, int):
            return StaticDim(dim=self.dim, size=self.size + other)
        if not isinstance(other, StaticDim):
            return NotImplemented
        if self.dim is not other.dim:
            raise TypeError(
                f"Cannot add StaticDims with different base Dims: "
                f"{self.dim.dim_name} vs {other.dim.dim_name}"
            )
        return StaticDim(dim=self.dim, size=self.size + other.size)

    def __radd__(self, other: int) -> "StaticDim":
        """Right add - handles int + StaticDim."""
        if isinstance(other, int):
            return StaticDim(dim=self.dim, size=self.size + other)
        return NotImplemented

    def fold(self, stride: int) -> "StaticFold":
        """Create a StaticFold of this dimension with the given stride.

        Args:
            stride: The fold stride.

        Returns:
            A StaticFold object representing this dimension folded by the given stride.

        Raises:
            ValueError: If the size is not divisible by the stride.

        Example:
            I = Dim()
            i = I(16)
            i8 = i.fold(8)  # StaticFold of i with stride 8
        """
        # pylint: disable-next=import-outside-toplevel  # circular import
        from .fold import StaticFold

        # Reuse the Dim's fold cache to ensure consistent Fold objects
        fold = self.dim.fold(stride)
        if self.size % stride != 0:
            raise ValueError(
                f"Cannot fold dimension of size {self.size} by stride {stride}. "
                f"Use fold_up() if you want to round up."
            )
        size_fold = self.size // stride
        return StaticFold(fold=fold, size=size_fold)

    def fold_up(self, stride: int) -> "StaticFold":
        """Create a StaticFold of this dimension with the given stride, rounding up.

        Like fold(), but uses ceiling division so the size doesn't need to be
        divisible by the stride. The resulting size is rounded up.

        Args:
            stride: The fold stride.

        Returns:
            A StaticFold object representing this dimension folded by the given stride,
            with size rounded up if not evenly divisible.

        Example:
            I = Dim()
            i = I(17)
            i8 = i.fold_up(8)  # StaticFold with size 3 (ceil(17/8) = 3)
        """
        # pylint: disable-next=import-outside-toplevel  # circular import
        from .fold import StaticFold

        # Reuse the Dim's fold cache to ensure consistent Fold objects
        fold = self.dim.fold(stride)
        size_fold = divup(self.size, stride)
        return StaticFold(fold=fold, size=size_fold)

    def __truediv__(self, stride: int) -> "StaticFold":
        """Create a StaticFold using the / operator.

        Equivalent to self.fold(stride).

        Example:
            i = I(128)
            i16 = i / 16  # StaticFold with size 8
        """
        if not isinstance(stride, int):
            return NotImplemented
        return self.fold(stride)

    def __mod__(self, modulo: int) -> "StaticDim":
        """Create a StaticDim with size equal to the modulo.

        This operation checks that the modulo divides the current size exactly,
        then returns a new StaticDim with the same base Dim and size equal to modulo.

        This is useful for specifying the "fine" dimension size after a fold,
        where the fold gives the coarse size and modulo gives the fine size.

        Args:
            modulo: The modulo value, which must evenly divide self.size.

        Returns:
            A new StaticDim with the same base Dim and size equal to modulo.

        Raises:
            ValueError: If modulo does not evenly divide self.size.

        Example:
            i = I(128)
            i_fine = i % 16  # StaticDim with size 16 (the remainder part)
            i_coarse = i / 16  # StaticFold with size 8 (the quotient part)
        """
        if not isinstance(modulo, int):
            return NotImplemented
        if self.size % modulo != 0:
            raise ValueError(f"Modulo {modulo} does not evenly divide size {self.size}.")
        return StaticDim(dim=self.dim, size=modulo)


@dataclass(frozen=True)
class Dim(GenSpecs):
    """CUDA Code generator for custom dimension classes.

    This class defines a named tensor dimension.

    Note: the spio.generators.generate() method will automatically detect
    all dimensions used in the generator specifications and generate the
    corresponding custom dimension classes. Normally the user will not
    need to use this class directly.

    Attributes:
        dim_name (str): The name of the dimension. If None, set via assignment in Generators.

    Example:
        # Create dimensions
        I = Dim()
        K = Dim()

        # Use with Dims to specify tensor dimensions
        Tensor("A", dtype.float, Dims(I(16), K(32)))
    """

    dim_name: str = None

    def __post_init__(self):
        """Normalize the dimension name to upper-case."""
        if self.dim_name is not None:
            object.__setattr__(self, "dim_name", self.dim_name.upper())
        # Initialize fold cache for caching Fold objects by stride
        object.__setattr__(self, "_fold_cache", {})

    def __call__(self, size: int) -> StaticDim:
        """Create a StaticDim with this dimension and the given size.

        Args:
            size: The size of the dimension.

        Returns:
            A StaticDim object representing this dimension with the given size.

        Example:
            I = Dim()
            I(16)  # StaticDim with dim=I and size=16
        """
        return StaticDim(dim=self, size=size)

    def fold(self, stride: int, name: str = None) -> "Fold":
        """Create a Fold of this dimension with the given stride.

        Args:
            stride: The fold stride.
            name: Optional name for the fold. If not provided, must be set
                  via Generators attribute assignment.

        Returns:
            A Fold object representing this dimension folded by the given stride.
            Repeated calls with the same stride return the same Fold object.

        Example:
            K = Dim()
            K8 = K.fold(8)  # Fold of K with stride 8
            K8 = K.fold(8, "K8")  # Fold with explicit name
        """
        # pylint: disable-next=import-outside-toplevel  # circular import
        from .fold import Fold

        # Cache Fold objects by stride for consistent identity
        cache = object.__getattribute__(self, "_fold_cache")
        if stride not in cache:
            cache[stride] = Fold(dim=self, stride=stride, fold_name=name)
        elif name is not None:
            # If a name is provided, update the cached Fold's name
            cached_fold = cache[stride]
            if cached_fold.fold_name is None:
                cached_fold._set_class_name(name)
        return cache[stride]

    def __truediv__(self, stride: int) -> "Fold":
        """Create a Fold using the / operator.

        Equivalent to self.fold(stride).

        Example:
            I = Dim()
            I16 = I / 16  # Fold with stride 16
        """
        if not isinstance(stride, int):
            return NotImplemented
        return self.fold(stride)

    def _set_class_name(self, name: str) -> None:
        """Set the dimension name from the Generators container attribute name."""
        object.__setattr__(self, "dim_name", name.upper())

    def get_class_name(self) -> str:
        """Return the class name, or None if dim_name is not set."""
        return self.dim_name

    @property
    def class_name(self) -> str:
        """Convert the dimension name to a dimension class name."""
        return _format_dim_class_name(self.dim_name)

    def generate(self):
        """Generate the C++ code for a dimension class using CRTP."""
        class_name = self.class_name
        return (
            f"struct {class_name} : spio::Dim<{class_name}> "
            f"{{ using spio::Dim<{class_name}>::Dim; }};\n"
        )

    @property
    def dim_names(self) -> Tuple[str,]:
        """Return the name of the dimension."""
        return (self.dim_name,)


def dim_name_to_dim_or_fold_class_name(name: str) -> str:
    """Convert a dimension name to a dimension or folded-dimension class name."""
    dim_class_name, dim_stride = _get_dim_name_and_stride(name)
    return _get_dim_or_fold_class_name(dim_class_name, dim_stride)


def _get_dim_or_fold_class_name(name: str, stride: int):
    dim_class_name = _format_dim_class_name(name)
    if stride is None:
        return dim_class_name
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
    return dim_name


def _format_fold_template_instance(dim_class_name: str, stride: int) -> str:
    return f"spio::Fold<{dim_class_name}, {stride}>"


def header() -> str:
    """Return a C++ statement that includes the spio dim header.

    The header implements the C++ spio::Dim base class from which the
    custom dimension classes inherit.
    """
    return '#include "spio/dim.h"'


BUILTIN_DIM_NAMES = ["LANE", "OFFSET"]

# Built-in dimensions that are predefined in the spio C++ headers.
# These don't generate code since they already exist in spio/dim.h.
LANE = Dim("LANE")
OFFSET = Dim("OFFSET")
