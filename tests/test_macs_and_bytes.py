from spio.kernels import (
    ConvSmallGroupKernel,
    ConvSmallGroupParams,
    ConvSmallGroupWgradKernel,
)


def test_conv_small_group_kernel_macs_and_bytes():
    p = ConvSmallGroupParams(N=4, C=128, H=64, W=32, R=3, S=5)
    assert ConvSmallGroupKernel.macs(p) == (p.N * p.C * p.P * p.Q) * (
        p.R * p.S * p.group_width
    )
    assert ConvSmallGroupKernel.bytes_read(p) == 2 * (
        (p.N * p.C * p.H * p.W) + (p.C * p.R * p.S * p.group_width)
    )
    assert ConvSmallGroupKernel.bytes_written(p) == 2 * (p.N * p.C * p.P * p.Q)

def test_conv_small_group_igrad_kernel_macs_and_bytes():
    p = ConvSmallGroupParams(N=4, C=128, H=64, W=32, R=3, S=5)
    assert ConvSmallGroupKernel.macs(p, igrad=True) == (p.N * p.C * p.P * p.Q) * (
        p.R * p.S * p.group_width
    )
    assert ConvSmallGroupKernel.bytes_read(p, igrad=True) == 2 * (
        (p.N * p.C * p.P * p.Q) + (p.C * p.R * p.S * p.group_width)
    )
    assert ConvSmallGroupKernel.bytes_written(p, igrad=True) == 2 * (p.N * p.C * p.H * p.W)


def test_conv_small_group_wgrad_kernel_macs_and_bytes():
    p = ConvSmallGroupParams(N=4, C=128, H=64, W=32, R=3, S=5)
    assert ConvSmallGroupWgradKernel.macs(p) == (p.N * p.C * p.P * p.Q) * (
        p.R * p.S * p.group_width
    )
    assert ConvSmallGroupWgradKernel.bytes_read(p) == 2 * (
        (p.N * p.C * p.H * p.W) + (p.N * p.C * p.P * p.Q)
    )
    assert ConvSmallGroupWgradKernel.bytes_written(p) == 2 * (
        p.C * p.R * p.S * p.group_width
    )
