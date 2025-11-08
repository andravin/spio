"""Tests for the compute statistics of the conv2d kernel."""

from spio.kernels import (
    Conv2dGw8Params,
    Conv2dStats,
)


def test_conv2d_stats_fprop():
    """Test the statistics for the forward pass of the conv2d kernel."""
    params = Conv2dGw8Params(n=16, h=32, w=64, c=128, r=3, s=3)
    stats = Conv2dStats(params, output_names="output")
    read = (16 * 32 * 64 * 128 + 128 * 8 * 3 * 3) * 2
    written = 16 * 32 * 64 * 128 * 2
    assert stats.bytes_read == read
    assert stats.bytes_written == written
    assert stats.bytes == read + written
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3
    assert stats.accumulation_depths == [8 * 3 * 3]


def test_conv2d_stats_input_grad():
    """Test the statistics for the input gradient of the conv2d kernel."""
    params = Conv2dGw8Params(n=16, h=32, w=64, c=128, r=3, s=3)
    stats = Conv2dStats(params, output_names="grad_input")
    read = (16 * 32 * 64 * 128 + 128 * 8 * 3 * 3) * 2
    written = 16 * 32 * 64 * 128 * 2
    assert stats.bytes_read == read
    assert stats.bytes_written == written
    assert stats.bytes == read + written
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3
    assert stats.accumulation_depths == [8 * 3 * 3]


def test_conv2d_stats_weight_grad():
    """Test the statistics for the weight gradient of the conv2d kernel."""
    params = Conv2dGw8Params(n=16, h=32, w=64, c=128, r=3, s=3)
    stats = Conv2dStats(params, output_names="grad_weight")
    read = (16 * 32 * 64 * 128 + 16 * 32 * 64 * 128) * 2
    written = 128 * 8 * 3 * 3 * 2
    assert stats.bytes_read == read
    assert stats.bytes_written == written
    assert stats.bytes == read + written
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3
    assert stats.accumulation_depths == [16 * 32 * 64]


def test_conv2d_stats_bias_grad():
    """Test the statistics for the bias gradient of the conv2d kernel."""
    params = Conv2dGw8Params(n=16, h=32, w=64, c=128, r=3, s=3)
    stats = Conv2dStats(params, output_names="grad_bias")
    assert stats.accumulation_depths == [16 * 32 * 64]
