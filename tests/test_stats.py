"""Tests for the compute statistics of the conv2d kernel."""

from spio.kernels.conv2d_gw8_kernel import conv2d_gw8_kernel_factory


def test_conv2d_stats_fprop():
    """Test the statistics for the forward pass of the conv2d kernel."""
    params = conv2d_gw8_kernel_factory.params_cls(N=16, H=32, W=64, C=128, R=3, S=3)
    stats = conv2d_gw8_kernel_factory.stats_cls(params, output_names="output")
    read = (16 * 32 * 64 * 128 + 128 * 8 * 3 * 3) * 2
    written = 16 * 32 * 64 * 128 * 2
    assert stats.bytes_read == read
    assert stats.bytes_written == written
    assert stats.bytes == read + written
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3
    assert stats.accumulation_depths == [8 * 3 * 3]


def test_conv2d_stats_input_grad():
    """Test the statistics for the input gradient of the conv2d kernel."""
    params = conv2d_gw8_kernel_factory.params_cls(N=16, H=32, W=64, C=128, R=3, S=3)
    stats = conv2d_gw8_kernel_factory.stats_cls(params, output_names="grad_input")
    read = (16 * 32 * 64 * 128 + 128 * 8 * 3 * 3) * 2
    written = 16 * 32 * 64 * 128 * 2
    assert stats.bytes_read == read
    assert stats.bytes_written == written
    assert stats.bytes == read + written
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3
    assert stats.accumulation_depths == [8 * 3 * 3]


def test_conv2d_stats_weight_grad():
    """Test the statistics for the weight gradient of the conv2d kernel."""
    params = conv2d_gw8_kernel_factory.params_cls(N=16, H=32, W=64, C=128, R=3, S=3)
    stats = conv2d_gw8_kernel_factory.stats_cls(params, output_names="grad_weight")
    read = (16 * 32 * 64 * 128 + 16 * 32 * 64 * 128) * 2
    written = 128 * 8 * 3 * 3 * 2
    assert stats.bytes_read == read
    assert stats.bytes_written == written
    assert stats.bytes == read + written
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3
    assert stats.accumulation_depths == [16 * 32 * 64]


def test_conv2d_stats_bias_grad():
    """Test the statistics for the bias gradient of the conv2d kernel."""
    params = conv2d_gw8_kernel_factory.params_cls(N=16, H=32, W=64, C=128, R=3, S=3)
    stats = conv2d_gw8_kernel_factory.stats_cls(params, output_names="grad_bias")
    assert stats.accumulation_depths == [16 * 32 * 64]
