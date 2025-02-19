"""Test the cuda driver interface."""

from spio.cuda import driver

MB = 1024 * 1024

DEVICE_ATTRIBUTES = {
    "NVIDIA GeForce RTX 4090": driver.DeviceAttributes(
        multiprocessor_count=128, l2_cache_size=72 * MB
    )
}


def test_get_multiprocessor_count():
    """Test the get_multiprocessor_count function."""
    sm_count = driver.get_multiprocessor_count()
    assert sm_count > 0
    assert sm_count == driver.get_device_attributes().multiprocessor_count
    attributes = DEVICE_ATTRIBUTES.get(driver.get_device_name())
    if attributes is not None:
        assert sm_count == attributes.multiprocessor_count


def test_get_l2_cache_size():
    """Test the get_l2_cache_size function."""
    l2_cache_size = driver.get_l2_cache_size()
    assert l2_cache_size > 0
    assert l2_cache_size == driver.get_device_attributes().l2_cache_size
    attributes = DEVICE_ATTRIBUTES.get(driver.get_device_name())
    if attributes is not None:
        assert l2_cache_size == attributes.l2_cache_size


def test_get_device_name():
    """Test the get_device_name function."""
    device_name = driver.get_device_name()
    assert len(device_name) > 0
    assert device_name == driver.get_device_attributes().name
