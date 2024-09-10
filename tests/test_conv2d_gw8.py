import pytest
import random
import os
from typing import List

from spio.runner import (
    run_kernel_test,
    run_grad_kernel_test,
    run_function_test,
    run_grad_function_test,
    run_layer_test,
)
from spio.kernels import Conv2dGw8Kernel, Conv2dGw8WgradKernel, Conv2dGw8Params
from spio.functional import conv2d_gw8
from spio.layers import Conv2dGw8

# Accelerate tests by limiting the number of parameters tested
#
# Each test will randomly sample from the available test parameters.
# Set SPIO_MAX_TEST_SAMPLES <= 0 to run all test cases.
DEFAULT_MAX_TEST_PARAMS = 10
MAX_TEST_PARAMS = int(
    os.environ.get("SPIO_MAX_TEST_PARAMS", f"{DEFAULT_MAX_TEST_PARAMS}")
)


def test_kernel_conv2d_gw8_sanity():
    params = Conv2dGw8Params(N=4, C=64, H=16, W=32, padding=1, R=3, S=3, has_bias=True)
    run_kernel_test(Conv2dGw8Kernel, params)


def test_kernel_conv2d_gw8_wgrad_sanity():
    params = Conv2dGw8Params(N=4, C=64, H=16, W=32, padding=1, R=3, S=3)
    run_grad_kernel_test(Conv2dGw8WgradKernel, params)


def test_functional_conv2d_gw8_grad_sanity():
    params = Conv2dGw8Params(
        N=1, C=128, H=32, W=32, padding=(2, 0), R=4, S=1, has_bias=True
    )
    run_grad_function_test(conv2d_gw8, params)


def _random_sample_test_params(max_samples=MAX_TEST_PARAMS) -> List[Conv2dGw8Params]:
    if max_samples <= 0:
        return _get_test_params_with_and_without_bias()
    else:
        return random.sample(_get_test_params_with_and_without_bias(), max_samples)


def _get_test_params_with_and_without_bias():
    return _get_test_params(has_bias=True) + _get_test_params(has_bias=False)


def _get_test_params(has_bias=False) -> List[Conv2dGw8Params]:
    kwargs = {"has_bias": has_bias}
    Params = Conv2dGw8Params

    model_tests = [
        Params(N=64, C=672, H=24, W=24, padding=1, R=3, S=3, **kwargs),
        Params(N=256, C=32, H=112, W=112, padding=1, R=3, S=3, **kwargs),
        Params(N=256, C=144, H=56, W=56, padding=1, R=3, S=3, **kwargs),
        Params(N=256, C=240, H=28, W=28, padding=2, R=5, S=5, **kwargs),
    ]

    more_model_tests = [
        Conv2dGw8Params(
            N=128, C=1024, H=7, W=7, padding=(1, 1), R=3, S=3, has_bias=True
        ),
        Conv2dGw8Params(
            N=128, C=1248, H=8, W=8, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=1248, H=8, W=8, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=128, H=56, W=56, padding=(1, 1), R=3, S=3, has_bias=True
        ),
        Conv2dGw8Params(
            N=128, C=1392, H=9, W=9, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=1392, H=9, W=9, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=144, H=64, W=64, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=16, H=128, W=128, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=192, H=72, W=72, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=2112, H=8, W=8, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=2304, H=9, W=9, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=24, H=144, W=144, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=256, H=28, W=28, padding=(1, 1), R=3, S=3, has_bias=True
        ),
        Conv2dGw8Params(
            N=128, C=288, H=32, W=32, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=288, H=36, W=36, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=32, H=128, W=128, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=40, H=144, W=144, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=512, H=14, W=14, padding=(1, 1), R=3, S=3, has_bias=True
        ),
        Conv2dGw8Params(
            N=128, C=528, H=16, W=16, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=528, H=16, W=16, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=576, H=18, W=18, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=576, H=18, W=18, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=720, H=16, W=16, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=128, C=816, H=18, W=18, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=256, C=1152, H=7, W=7, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=256, C=1152, H=7, W=7, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=256, C=144, H=56, W=56, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=256, C=16, H=112, W=112, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=256, C=1920, H=7, W=7, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=256, C=240, H=28, W=28, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=256, C=32, H=112, W=112, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=256, C=480, H=14, W=14, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=256, C=480, H=14, W=14, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=256, C=672, H=14, W=14, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=32, C=1056, H=28, W=28, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=32, C=1824, H=14, W=14, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=32, C=1824, H=14, W=14, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=32, C=240, H=112, W=112, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=32, C=24, H=224, W=224, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=32, C=3072, H=14, W=14, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=32, C=384, H=56, W=56, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=32, C=48, H=224, W=224, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=32, C=768, H=28, W=28, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=32, C=768, H=28, W=28, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=64, C=1024, H=7, W=7, padding=(1, 1), R=3, S=3, has_bias=True
        ),
        Conv2dGw8Params(
            N=64, C=128, H=56, W=56, padding=(1, 1), R=3, S=3, has_bias=True
        ),
        Conv2dGw8Params(
            N=64, C=1632, H=10, W=10, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=64, C=1632, H=10, W=10, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=64, C=192, H=80, W=80, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=64, C=24, H=160, W=160, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=64, C=256, H=28, W=28, padding=(1, 1), R=3, S=3, has_bias=True
        ),
        Conv2dGw8Params(
            N=64, C=2688, H=10, W=10, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=64, C=336, H=40, W=40, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=64, C=48, H=160, W=160, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=64, C=512, H=14, W=14, padding=(1, 1), R=3, S=3, has_bias=True
        ),
        Conv2dGw8Params(
            N=64, C=672, H=20, W=20, padding=(1, 1), R=3, S=3, has_bias=False
        ),
        Conv2dGw8Params(
            N=64, C=672, H=20, W=20, padding=(2, 2), R=5, S=5, has_bias=False
        ),
        Conv2dGw8Params(
            N=64, C=960, H=20, W=20, padding=(2, 2), R=5, S=5, has_bias=False
        ),
    ]

    conv_rxs_tests = [
        Params(N=1, C=128, H=32, W=32, R=r, S=s, padding=(r // 2, s // 2), **kwargs)
        for r in range(1, 6)
        for s in range(1, 6)
    ]

    padding_tests = [
        Params(N=1, C=128, H=64, W=64, padding=0, **kwargs),
        Params(N=4, C=128, H=20, W=19, padding=0, **kwargs),
        Params(N=1, C=128, H=64, W=64, padding=2, **kwargs),
        Params(N=1, C=128, H=64, W=64, padding=7, **kwargs),
    ]

    min_groups = 1
    max_groups = 16
    group_tests = [
        Params(N=4, C=groups * 8, H=32, W=32, **kwargs)
        for groups in range(min_groups, max_groups + 1)
    ]

    misc_tests = [
        Params(N=6, C=128, H=49, W=33, **kwargs),
        Params(N=6, C=128, H=6, W=5, **kwargs),
        Params(N=128, C=1024, H=7, W=7, **kwargs),
        Params(N=128, C=64, H=64, W=64, **kwargs),
        Params(N=128, C=128, H=64, W=64, **kwargs),
        Params(N=128, C=256, H=64, W=64, **kwargs),
        Params(N=3, C=128, H=16, W=16, **kwargs),
        Params(N=4, C=128, H=32, W=16, **kwargs),
        Params(N=5, C=128, H=16, W=32, **kwargs),
        Params(N=6, C=128, H=48, W=32, **kwargs),
    ]

    return (
        more_model_tests
        + model_tests
        + conv_rxs_tests
        + padding_tests
        + group_tests
        + misc_tests
    )


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_kernel_conv2d_gw8(params: Conv2dGw8Params):
    run_kernel_test(Conv2dGw8Kernel, params)


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_kernel_conv2d_gw8_wgrad(params: Conv2dGw8Params):
    run_grad_kernel_test(Conv2dGw8WgradKernel, params)


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_kernel_conv2d_gw8_igrad(params: Conv2dGw8Params):
    run_grad_kernel_test(Conv2dGw8Kernel, params, igrad=True)


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_functional_conv2d_gw8(params: Conv2dGw8Params):
    run_function_test(conv2d_gw8, params)


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_functional_conv2d_gw8_grad(params: Conv2dGw8Params):
    """NOTE this test failed when it was run isolation due to an unknown race condition that caused a CUDA_ERROR_INVALID_CONTEXT.
    If this test was run after the tests that precede it, there is no error.
    I added a workaround in the benchmark function to synchronize the CUDA context before loading the kernel.
    This hack seems to have fixed the issue.
    """
    run_grad_function_test(conv2d_gw8, params)


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_conv2d_gw8_layer(params: Conv2dGw8Params):
    run_layer_test(Conv2dGw8, params)
