"""Define the dtype enumeration."""
from enum import Enum


class _DataType:
    """A class to represent a data type."""

    def __init__(self, name: str, size: int):
        """Initialize the DataType object with the given name and size in bytes."""
        self.name = name
        self.size = size


class dtype(Enum):
    """Data type enumeration.

    Use one of these constants for any spio function that requests a data type.
    Data types are used to specify the type of data stored in a tensor.
    """

    float = _DataType("float", 4)
    float2 = _DataType("float2", 8)
    float4 = _DataType("float4", 16)
    unsigned = _DataType("unsigned", 4)
    uint2 = _DataType("uint2", 8)
    uint4 = _DataType("uint4", 16)
    half = _DataType("__half", 2)
    half2 = _DataType("__half2", 4)
    int32 = _DataType("int", 4)
