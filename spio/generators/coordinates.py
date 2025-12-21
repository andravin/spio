"""Code generator for Coordinates objects."""

from dataclasses import dataclass
from typing import Union

from .gen_specs import GenSpecs
from .dim import Dim
from .fold import Fold


def _dim_or_fold_name(dim: Union[Dim, Fold]) -> str:
    """Return the name of the dimension or fold."""
    return dim.dim_name if isinstance(dim, Dim) else dim.fold_name


@dataclass(frozen=True)
class _CoordinatesSpec(GenSpecs):
    """CUDA Code generator for Coordinates objects (base dataclass).

    Use the Coordinates subclass for a more flexible constructor that accepts varargs.

    Attributes:
        dims: A list of dimension and/or fold generators that compose this Coordinates.
        coord_name: The name of the Coordinates class (optional with Generators).
    """

    dims: list[Union[Dim, Fold]]
    coord_name: str = None

    def __post_init__(self):
        """Validate that dims types."""
        for dim in self.dims:
            if not isinstance(dim, (Dim, Fold)):
                raise TypeError(
                    f"Coordinates dims must be Dim or Fold instances, got {type(dim)}"
                )

    def _set_class_name(self, name: str) -> None:
        """Set the coordinate name for this Coordinates.

        Called by the Generators container when the coordinates is assigned to an attribute.
        """
        object.__setattr__(self, "coord_name", name)

    def get_class_name(self) -> str:
        """Return the coordinate name, or None if not set."""
        return self.coord_name

    def generate(self):
        dim_class_names = [_dim_or_fold_name(dim) for dim in self.dims]
        init_list = ", ".join([name + "()" for name in dim_class_names])
        return f"""
auto {self.coord_name}() {{
    return make_coordinates({init_list});
}}
"""

    def used_generators(self) -> list[GenSpecs]:
        """Return a list of generator class-names used by this generator.

        This is used to find any generators that need to be assigned class names automatically.
        Subclasses should override this method if they use other generators.
        """
        return self.dims


class Coordinates(_CoordinatesSpec):
    """CUDA Code generator for Coordinates objects.

    This class defines a function that makes a Coordinates object composed of multiple
    dimensions and/or folded dimensions.

    When used with the Generators container, coord_name can be omitted and will
    be set from the attribute name.

    This is most useful when the Fold or Dim objects have initializing constructors.

    For example:

        g.block_i = Fold("i", 128, init=BuiltIn.BLOCK_IDX_Y)
        g.block_j = Fold("j", 128, init=BuiltIn.BLOCK_IDX_X)
        g.BlockIdx = Coordinates(g.block_i, g.block_j)
    """

    def __new__(cls, *args, coord_name: str = None):
        dims = list(args)
        instance = object.__new__(cls)
        object.__setattr__(instance, "dims", dims)
        object.__setattr__(instance, "coord_name", coord_name)
        return instance

    def __init__(
        self, *_args, coord_name: str = None
    ):  # pylint: disable=super-init-not-called
        # Attributes set in __new__; calling parent __init__ would fail on frozen dataclass
        self.__post_init__()
