"""Test the dtype enumeration"""

import spio.generators as gen


def test_dtype_sizes():
    """Test the sizes of the data types."""
    assert gen.dtype.float.value.size == 4
    assert gen.dtype.float2.value.size == 8
    assert gen.dtype.float4.value.size == 16
    assert gen.dtype.half.value.size == 2
    assert gen.dtype.half2.value.size == 4
    assert gen.dtype.unsigned.value.size == 4
    assert gen.dtype.uint2.value.size == 8
    assert gen.dtype.uint4.value.size == 16
    assert gen.dtype.int32.value.size == 4
