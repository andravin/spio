"""Define the dtype enumeration."""

from enum import Enum


class _ScalarDataType:
    """A class to represent a scalar (non-vector) data type."""

    def __init__(self, name: str, size: int):
        """Initialize the ScalarDataType with the given name and size in bytes."""
        self.name = name
        self.size = size

    def __repr__(self) -> str:
        return f"_ScalarDataType({self.name!r}, {self.size})"


class scalar_dtype(Enum):
    """Scalar data type enumeration.

    These are the fundamental scalar types that vector types are built from.
    """

    float = _ScalarDataType("float", 4)
    unsigned = _ScalarDataType("unsigned", 4)
    half = _ScalarDataType("__half", 2)
    int32 = _ScalarDataType("int", 4)


class _DataType:
    """A class to represent a data type (scalar or vector)."""

    def __init__(self, name: str, scalar: scalar_dtype, veclen: int):
        """Initialize the DataType with the given name, scalar type, and vector length."""
        self.name = name
        self.scalar_dtype = scalar
        self.veclen = veclen

    @property
    def size(self) -> int:
        """Return the size in bytes of this data type."""
        return self.scalar_dtype.value.size * self.veclen

    def __repr__(self) -> str:
        return f"_DataType({self.name!r}, {self.scalar_dtype}, {self.veclen})"


class dtype(Enum):
    """Data type enumeration.

    Use one of these constants for any spio function that requests a data type.
    Data types are used to specify the type of data stored in a tensor.
    """

    # Float types
    float = _DataType("float", scalar_dtype.float, 1)
    float2 = _DataType("float2", scalar_dtype.float, 2)
    float4 = _DataType("float4", scalar_dtype.float, 4)

    # Unsigned types
    unsigned = _DataType("unsigned", scalar_dtype.unsigned, 1)
    uint2 = _DataType("uint2", scalar_dtype.unsigned, 2)
    uint4 = _DataType("uint4", scalar_dtype.unsigned, 4)

    # Half types
    half = _DataType("__half", scalar_dtype.half, 1)
    half2 = _DataType("__half2", scalar_dtype.half, 2)
    half8 = _DataType("uint4", scalar_dtype.half, 8)  # half8 uses uint4 storage

    # Integer types
    int32 = _DataType("int", scalar_dtype.int32, 1)


def get_dtype_veclen(dtype_value: dtype) -> int:
    """Return the vector length of the given data type.

    The vector length is the number of scalar elements in the data type.
    For example, float4 has a vector length of 4, while float has a vector length of 1.

    Args:
        dtype_value: The data type to get the vector length for.

    Returns:
        The vector length of the data type.
    """
    return dtype_value.value.veclen


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
    target_scalar = dtype_value.value.scalar_dtype

    # Find the dtype with matching scalar type and vector length
    for dt in dtype:
        if dt.value.scalar_dtype == target_scalar and dt.value.veclen == veclen:
            return dt

    raise ValueError(f"No {target_scalar.name} dtype with vector length {veclen}")
