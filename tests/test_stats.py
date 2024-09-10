from spio.kernels.conv2d_gw8_kernel import Conv2dGw8Kernel

def test_conv2d_stats():
    params = Conv2dGw8Kernel.Params(N=16, H=32, W=64, C=128, R=3, S=3)
    stats = Conv2dGw8Kernel.Stats(params, output_names="output")
    assert stats.bytes_read == (16 * 32 * 64 * 128 + 128 * 8 * 3 * 3) * 2
    assert stats.bytes_written == 16 * 32 * 64 * 128 * 2
    assert stats.macs == (16 * 32 * 64 * 128) * 8 * 3 * 3

# def test_conv_small_group_kernel_macs_and_bytes():
#     p = Conv2dGw8Params(N=4, C=128, H=64, W=32, R=3, S=5)
#     assert Conv2dGw8Kernel.macs(p) == (p.N * p.C * p.P * p.Q) * (
#         p.R * p.S * p.group_width
#     )
#     assert Conv2dGw8Kernel.bytes_read(p) == 2 * (
#         (p.N * p.C * p.H * p.W) + (p.C * p.R * p.S * p.group_width)
#     )
#     assert Conv2dGw8Kernel.bytes_written(p) == 2 * (p.N * p.C * p.P * p.Q)

# def test_conv_small_group_igrad_kernel_macs_and_bytes():
#     p = Conv2dGw8Params(N=4, C=128, H=64, W=32, R=3, S=5)
#     assert Conv2dGw8Kernel.macs(p, igrad=True) == (p.N * p.C * p.P * p.Q) * (
#         p.R * p.S * p.group_width
#     )
#     assert Conv2dGw8Kernel.bytes_read(p, igrad=True) == 2 * (
#         (p.N * p.C * p.P * p.Q) + (p.C * p.R * p.S * p.group_width)
#     )
#     assert Conv2dGw8Kernel.bytes_written(p, igrad=True) == 2 * (p.N * p.C * p.H * p.W)


# def test_conv_small_group_wgrad_kernel_macs_and_bytes():
#     p = Conv2dGw8Params(N=4, C=128, H=64, W=32, R=3, S=5)
#     assert Conv2dGw8WgradKernel.macs(p) == (p.N * p.C * p.P * p.Q) * (
#         p.R * p.S * p.group_width
#     )
#     assert Conv2dGw8WgradKernel.bytes_read(p) == 2 * (
#         (p.N * p.C * p.H * p.W) + (p.N * p.C * p.P * p.Q)
#     )
#     assert Conv2dGw8WgradKernel.bytes_written(p) == 2 * (
#         p.C * p.R * p.S * p.group_width
#     )
