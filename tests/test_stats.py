from spio.kernels.conv2d_gw8_kernel import Conv2dGw8Kernel


def test_conv2d_stats_fprop():
    params = Conv2dGw8Kernel.Params(N=16, H=32, W=64, C=128, R=3, S=3)
    stats = Conv2dGw8Kernel.Stats(params, output_names="output")
    read = (16 * 32 * 64 * 128 + 128 * 8 * 3 * 3) * 2
    written = 16 * 32 * 64 * 128 * 2
    assert stats.bytes_read == read
    assert stats.bytes_written == written
    assert stats.bytes == read + written
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3


def test_conv2d_stats_input_grad():
    params = Conv2dGw8Kernel.Params(N=16, H=32, W=64, C=128, R=3, S=3)
    stats = Conv2dGw8Kernel.Stats(params, output_names="grad_input")
    read = (16 * 32 * 64 * 128 + 128 * 8 * 3 * 3) * 2
    written = 16 * 32 * 64 * 128 * 2
    assert stats.bytes_read == read
    assert stats.bytes_written == written
    assert stats.bytes == read + written
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3


def test_conv2d_stats_weight_grad():
    params = Conv2dGw8Kernel.Params(N=16, H=32, W=64, C=128, R=3, S=3)
    stats = Conv2dGw8Kernel.Stats(params, output_names="grad_weight")
    read = (16 * 32 * 64 * 128 + 16 * 32 * 64 * 128) * 2
    written = 128 * 8 * 3 * 3 * 2
    assert stats.bytes_read == read
    assert stats.bytes_written == written
    assert stats.bytes == read + written
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3
