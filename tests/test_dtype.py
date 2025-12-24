"""Test the dtype enumeration"""

from spio.generators import dtype, get_dtype_veclen


def test_dtype_sizes():
    """Test the sizes of the data types."""
    assert dtype.float.value.size == 4
    assert dtype.float2.value.size == 8
    assert dtype.float4.value.size == 16
    assert dtype.half.value.size == 2
    assert dtype.half2.value.size == 4
    assert dtype.half8.value.size == 16
    assert dtype.unsigned.value.size == 4
    assert dtype.uint2.value.size == 8
    assert dtype.uint4.value.size == 16
    assert dtype.int32.value.size == 4


def test_dtype_veclen():
    """Test the vector lengths of the data types."""
    assert get_dtype_veclen(dtype.float) == 1
    assert get_dtype_veclen(dtype.float2) == 2
    assert get_dtype_veclen(dtype.float4) == 4
    assert get_dtype_veclen(dtype.half) == 1
    assert get_dtype_veclen(dtype.half2) == 2
    assert get_dtype_veclen(dtype.half8) == 8
    assert get_dtype_veclen(dtype.unsigned) == 1
    assert get_dtype_veclen(dtype.uint2) == 2
    assert get_dtype_veclen(dtype.uint4) == 4
    assert get_dtype_veclen(dtype.int32) == 1
