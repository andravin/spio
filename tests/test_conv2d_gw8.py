import torch

from spio.functional import conv2d_gw8

from spio.util import (
    run_function_test,
    run_function_tests,
    run_grad_function_test,
    run_grad_function_tests,
    run_kernel_tests,
)
from spio.kernels import Conv2dGw8Kernel, Conv2dGw8Params

Params = Conv2dGw8Params


def test_one_conv2d_gw8_3x3():
    run_function_test(conv2d_gw8, Params(N=128, C=128, H=64, W=64))


def test_conv2d_gw8_5x5():
    run_function_test(
        conv2d_gw8,
        Params(N=128, C=128, H=64, W=64, R=5, S=5, padding=2),
    )


def test_conv2d_gw8_grad_3x3():
    run_grad_function_test(conv2d_gw8, Params(N=128, C=128, H=64, W=64))


def test_conv2d_gw8_function():
    run_function_tests(conv2d_gw8, _get_test_params())


def test_conv2d_gw8_kernel():
    run_kernel_tests(Conv2dGw8Kernel, _get_test_params())


def test_conv2d_gw8_grad():
    run_grad_function_tests(conv2d_gw8, _get_test_params())


def _get_test_params():
    model_tests = [
        Params(N=64, C=672, H=24, W=24, padding=1, R=3, S=3),
        Params(N=256, C=32, H=112, W=112, padding=1, R=3, S=3),
        Params(N=256, C=144, H=56, W=56, padding=1, R=3, S=3),
        Params(N=256, C=240, H=28, W=28, padding=2, R=5, S=5),
    ]
    conv_rxs_tests = [
        Params(N=1, C=128, H=32, W=32, R=r, S=s, padding=(r // 2, s // 2))
        for r in range(1, 6)
        for s in range(1, 6)
    ]

    padding_tests = [
        Params(N=1, C=128, H=64, W=64, padding=0),
        Params(N=4, C=128, H=20, W=19, padding=0),
        Params(N=1, C=128, H=64, W=64, padding=2),
        Params(N=1, C=128, H=64, W=64, padding=7),
    ]

    min_groups = 1
    max_groups = 16
    group_tests = [
        Params(N=4, C=groups * 8, H=32, W=32)
        for groups in range(min_groups, max_groups + 1)
    ]

    misc_tests = [
        Params(N=6, C=128, H=49, W=33),
        Params(N=6, C=128, H=6, W=5),
        Params(N=128, C=1024, H=7, W=7),
        Params(N=128, C=64, H=64, W=64),
        Params(N=128, C=128, H=64, W=64),
        Params(N=128, C=256, H=64, W=64),
        Params(N=3, C=128, H=16, W=16),
        Params(N=4, C=128, H=32, W=16),
        Params(N=5, C=128, H=16, W=32),
        Params(N=6, C=128, H=48, W=32),
    ]

    all_tests = model_tests + conv_rxs_tests + padding_tests + group_tests + misc_tests
    return all_tests
