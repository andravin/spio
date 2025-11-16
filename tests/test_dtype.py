"""Test the dtype enumeration"""

from spio.generators import dtype


def test_dtype_sizes():
    """Test the sizes of the data types."""
    assert dtype.float.value.size == 4
    assert dtype.float2.value.size == 8
    assert dtype.float4.value.size == 16
    assert dtype.half.value.size == 2
    assert dtype.half2.value.size == 4
    assert dtype.unsigned.value.size == 4
    assert dtype.uint2.value.size == 8
    assert dtype.uint4.value.size == 16
    assert dtype.int32.value.size == 4
