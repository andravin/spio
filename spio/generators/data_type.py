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
    half8 = _DataType("uint4", 16)  # half8 is a synonym for uint4
    half = _DataType("__half", 2)
    half2 = _DataType("__half2", 4)
    int32 = _DataType("int", 4)


def get_dtype_veclen(dtype_value: dtype) -> int:
    """Return the vector length of the given data type.

    The vector length is the number of scalar elements in the data type.
    For example, float4 has a vector length of 4, while float has a vector length of 1.

    Args:
        dtype_value: The data type to get the vector length for.

    Returns:
        The vector length of the data type.
    """
    if dtype_value in {dtype.float, dtype.unsigned, dtype.int32, dtype.half}:
        return 1
    elif dtype_value in {dtype.float2, dtype.uint2, dtype.half2}:
        return 2
    elif dtype_value in {dtype.float4, dtype.uint4}:
        return 4
    elif dtype_value == dtype.half8:
        return 8
    else:
        raise ValueError(f"Vector length for data type {dtype_value} not defined.")


def get_dtype_with_veclen(dtype_value: dtype, veclen: int) -> dtype:
    """Return the dtype with the given vector length, preserving the base type.

    Args:
        dtype_value: The source data type (used to determine the base scalar type).
        veclen: The desired vector length.

    Returns:
        The dtype with the specified vector length.

    Raises:
        ValueError: If no dtype exists for the given base type and vector length.
    """
    # Determine the base scalar type
    if dtype_value in {dtype.half, dtype.half2, dtype.half8}:
        if veclen == 1:
            return dtype.half
        elif veclen == 2:
            return dtype.half2
        elif veclen == 8:
            return dtype.half8
        else:
            raise ValueError(f"No half dtype with vector length {veclen}")
    elif dtype_value in {dtype.float, dtype.float2, dtype.float4}:
        if veclen == 1:
            return dtype.float
        elif veclen == 2:
            return dtype.float2
        elif veclen == 4:
            return dtype.float4
        else:
            raise ValueError(f"No float dtype with vector length {veclen}")
    elif dtype_value in {dtype.unsigned, dtype.uint2, dtype.uint4}:
        if veclen == 1:
            return dtype.unsigned
        elif veclen == 2:
            return dtype.uint2
        elif veclen == 4:
            return dtype.uint4
        else:
            raise ValueError(f"No uint dtype with vector length {veclen}")
    else:
        raise ValueError(f"Cannot determine base type for {dtype_value}")
