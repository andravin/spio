"""Test the cuda driver interface."""

from spio.cuda import driver

KB = 1024
MB = KB * KB

# If the tested device is in this dictionary, the values of its attributes will be checked.
# Otherwise, only sanity tests will be performed.
DEVICE_ATTRIBUTES = {
    "NVIDIA GeForce RTX 4090": driver.DeviceAttributes(
        multiprocessor_count=128,
        l2_cache_size=72 * MB,
        name="NVIDIA GeForce RTX 4090",
        compute_capability=(8, 9),
        max_shared_memory_per_block_optin=99 * KB,
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
    attributes = DEVICE_ATTRIBUTES.get(device_name)
    if attributes is not None:
        assert device_name == attributes.name


def test_get_compute_capability():
    """Test the get_compute_capability function."""
    compute_capability = driver.get_compute_capability()
    assert isinstance(compute_capability, tuple)
    assert len(compute_capability) == 2
    assert all(isinstance(x, int) for x in compute_capability)
    assert compute_capability == driver.get_device_attributes().compute_capability
    attributes = DEVICE_ATTRIBUTES.get(driver.get_device_name())
    if attributes is not None:
        assert compute_capability == attributes.compute_capability


def test_get_max_shared_memory_per_block_optin():
    """Test the get_max_shared_memory_per_block_optin function."""
    max_shared_memory = driver.get_max_shared_memory_per_block_optin()
    assert max_shared_memory > 0
    assert (
        max_shared_memory
        == driver.get_device_attributes().max_shared_memory_per_block_optin
    )
    attributes = DEVICE_ATTRIBUTES.get(driver.get_device_name())
    if attributes is not None:
        assert max_shared_memory == attributes.max_shared_memory_per_block_optin
