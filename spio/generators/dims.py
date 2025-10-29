"""This file implements the Dims class."""

from typing import Dict, Union, Generator, Tuple

from .subindex_protocol import SubindexProtocol


class Dims:
    """A class to represent the dimensions of a tensor."""

    def __init__(self, **dims: Dict[str, Union[int, SubindexProtocol]]):
        """Initialize the Dims object with the given dimensions.

        Args:
            **dims: Keyword arguments representing the dimensions. Each dimension
                is specified as a name-value pair, where the name is a string
                and the value is an integer or a SubindexProtocol object.
        """
        self._dims = dims

    def items(self) -> Generator[Tuple[str, Union[int, SubindexProtocol]], None, None]:
        """Get the dimensions as a generator of (name, value) pairs."""
        return self._dims.items()

    def keys(self) -> Generator[str, None, None]:
        """Get the names of the dimensions."""
        return self._dims.keys()

    def values(self) -> Generator[Union[int, SubindexProtocol], None, None]:
        """Get the values of the dimensions."""
        return self._dims.values()

    def __getitem__(self, key) -> Union[int, SubindexProtocol]:
        """Get the value of a dimension by its name."""
        return self._dims[key]

    def __contains__(self, key) -> bool:
        """Check if a dimension is present in the Dims object."""
        return key in self._dims


class Strides:
    """A class to represent the strides of a tensor."""

    def __init__(self, **strides: Dict[str, int]):
        """Initialize the Strides object with the given strides.

        Args:
            **strides: Keyword arguments representing the strides. Each stride
                is specified as a name-value pair, where the name is a string
                and the value is an integer.
        """
        self._strides = strides

    def items(self) -> Generator[Tuple[str, int], None, None]:
        """Get the strides as a generator of (name, value) pairs."""
        return self._strides.items()

    def keys(self) -> Generator[str, None, None]:
        """Get the names of the strides."""
        return self._strides.keys()

    def values(self) -> Generator[int, None, None]:
        """Get the values of the strides."""
        return self._strides.values()

    def __getitem__(self, key) -> int:
        """Get the value of a stride by its name."""
        return self._strides[key]

    def __contains__(self, key) -> bool:
        """Check if a stride is present in the Strides object."""
        return key in self._strides


def compute_full_strides(
    dims: Dims,
    given_strides: Strides,
) -> Dict[str, int]:
    """Compute the full strides for the given dimensions."""
    if given_strides is None:
        given_strides = {}
    all_strides = {}
    stride = 1
    for name, value in reversed(dims.items()):
        if name in given_strides:
            stride = given_strides[name]
        all_strides[name] = stride
        # Compute the default stride of the next dimension.
        stride *= value
    return all_strides

