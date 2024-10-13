from spio.kernels.conv2d_gw8_kernel import conv2d_gw8_kernel_factory


def test_conv2d_stats_fprop():
    params = conv2d_gw8_kernel_factory.params_cls(N=16, H=32, W=64, C=128, R=3, S=3)
    stats = conv2d_gw8_kernel_factory.stats_cls(params, output_names="output")
    read = (16 * 32 * 64 * 128 + 128 * 8 * 3 * 3) * 2
    written = 16 * 32 * 64 * 128 * 2
    assert stats.bytes_read == read
    assert stats.bytes_written == written
    assert stats.bytes == read + written
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3


def test_conv2d_stats_input_grad():
    params = conv2d_gw8_kernel_factory.params_cls(N=16, H=32, W=64, C=128, R=3, S=3)
    stats = conv2d_gw8_kernel_factory.stats_cls(params, output_names="grad_input")
    read = (16 * 32 * 64 * 128 + 128 * 8 * 3 * 3) * 2
    written = 16 * 32 * 64 * 128 * 2
    assert stats.bytes_read == read
    assert stats.bytes_written == written
    assert stats.bytes == read + written
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3


def test_conv2d_stats_weight_grad():
    params = conv2d_gw8_kernel_factory.params_cls(N=16, H=32, W=64, C=128, R=3, S=3)
    stats = conv2d_gw8_kernel_factory.stats_cls(params, output_names="grad_weight")
    read = (16 * 32 * 64 * 128 + 16 * 32 * 64 * 128) * 2
    written = 128 * 8 * 3 * 3 * 2
    assert stats.bytes_read == read
    assert stats.bytes_written == written
    assert stats.bytes == read + written
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3
