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

    def __get_item__(self, key) -> Union[int, SubindexProtocol]:
        """Get the value of a dimension by its name."""
        return self._dims[key]
