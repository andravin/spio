"""Test the dtype enumeration"""

from spio.generators import dtype


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
    assert dtype.float.value.veclen == 1
    assert dtype.float2.value.veclen == 2
    assert dtype.float4.value.veclen == 4
    assert dtype.half.value.veclen == 1
    assert dtype.half2.value.veclen == 2
    assert dtype.half8.value.veclen == 8
    assert dtype.unsigned.value.veclen == 1
    assert dtype.uint2.value.veclen == 2
    assert dtype.uint4.value.veclen == 4
    assert dtype.int32.value.veclen == 1
