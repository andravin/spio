from spio.src_tests import run_kernel_test

from spio.kernels import (
    conv2d_gw8_kernel_factory,
    Conv2dGw8Params,
    Conv2dGw8Config,
)


def test_kernel_conv2d_gw8_troubleshoot_standalone():
    """Troubleshoot a kernel that failed in the function test.

    It also fails as a standalone kernel test; but passes when one of many kernel tests.
    This is symptomatic of an out-of-bounds memory access error.
    """
    config = Conv2dGw8Config(groups=6, block_p=8, block_n=2)
    params = Conv2dGw8Params(
        n=4,
        c=48,
        h=32,
        w=32,
        padding=1,
        r=3,
        s=3,
        has_bias=False,
        group_width=8,
        stride=1,
    )
    run_kernel_test(conv2d_gw8_kernel_factory, params, configs=[config])
