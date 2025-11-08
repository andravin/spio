"""Unit tests for the Conv2dGw8[Wgrad] kernels, function, and layers."""

from dataclasses import replace
import random
import os
from typing import List

import pytest

from spio.src_tests import (
    run_kernel_test,
    run_grad_kernel_test,
    run_function_test,
    run_grad_function_test,
    run_layer_test,
    run_opcheck_test,
)
from spio.kernels import (
    conv2d_gw8_kernel_factory,
    conv2d_gw8_wgrad_kernel_factory,
    Conv2dGw8Params,
    Conv2dGw8Config,
    Conv2dGw8WgradConfig,
)
from spio.functional import conv2d_gw8
from spio.layers import Conv2dGw8
from spio.util.load_parameter_set import load_parameter_set

# Accelerate tests by limiting the number of parameters tested
#
# Each test will randomly sample from the available test parameters.
# Set the environment variable SPIO_MAX_TEST_SAMPLES to a positive number
# to run that many tests or zero to run all.
DEFAULT_MAX_TEST_PARAMS = 10
MAX_TEST_PARAMS = int(
    os.environ.get("SPIO_MAX_TEST_PARAMS", f"{DEFAULT_MAX_TEST_PARAMS}")
)


def _random_sample_test_params(
    max_samples: int = MAX_TEST_PARAMS,
    allow_bias: bool = True,
    always_bias: bool = False,
) -> List[Conv2dGw8Params]:
    """Randomly sample test parameters from the available test parameters.

    Parameters
    ----------
    max_samples : int
        The maximum number of test parameters to sample.
        If max_samples <= 0, all test parameters are returned.

    Returns
    -------
    List[Conv2dGw8Params]
        A list of randomly sampled test parameters.
    """

    all_samples = []
    if allow_bias:
        all_samples += _get_test_params(has_bias=True)
    if not always_bias:
        all_samples += _get_test_params(has_bias=False)
    if max_samples <= 0:
        return all_samples
    return random.sample(all_samples, max_samples)


def _get_test_params(has_bias=False) -> List[Conv2dGw8Params]:
    kwargs = {"has_bias": has_bias}
    Params = Conv2dGw8Params

    model_tests = load_parameter_set(Params)
    model_tests = [replace(params, **kwargs) for params in model_tests]

    conv_rxs_tests = [
        Params(n=1, c=128, h=32, w=32, r=r, s=s, padding=(r // 2, s // 2), **kwargs)
        for r in range(1, 6)
        for s in range(1, 6)
    ]

    padding_tests = [
        Params(n=1, c=128, h=64, w=64, padding=0, **kwargs),
        Params(n=4, c=128, h=20, w=19, padding=0, **kwargs),
        Params(n=1, c=128, h=64, w=64, padding=2, **kwargs),
        Params(n=1, c=128, h=64, w=64, padding=7, **kwargs),
    ]

    min_groups = 1
    max_groups = 16
    group_tests = [
        Params(n=4, c=groups * 8, h=32, w=32, **kwargs)
        for groups in range(min_groups, max_groups + 1)
    ]

    misc_tests = [
        Params(n=6, c=128, h=49, w=33, **kwargs),
        Params(n=6, c=128, h=6, w=5, **kwargs),
        Params(n=128, c=1024, h=7, w=7, **kwargs),
        Params(n=128, c=64, h=64, w=64, **kwargs),
        Params(n=128, c=128, h=64, w=64, **kwargs),
        Params(n=128, c=256, h=64, w=64, **kwargs),
        Params(n=3, c=128, h=16, w=16, **kwargs),
        Params(n=4, c=128, h=32, w=16, **kwargs),
        Params(n=5, c=128, h=16, w=32, **kwargs),
        Params(n=6, c=128, h=48, w=32, **kwargs),
    ]

    return model_tests + conv_rxs_tests + padding_tests + group_tests + misc_tests


def test_kernel_conv2d_gw8_sanity():
    """Sanity test for the Conv2dGw8 kernel."""
    params = Conv2dGw8Params(n=4, c=64, h=16, w=32, padding=1, r=3, s=3, has_bias=True)
    run_kernel_test(conv2d_gw8_kernel_factory, params)


def test_kernel_conv2d_gw8_sanity_2():
    """Second sanity test for the Conv2dGw8 kernel."""
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


def test_kernel_conv2d_gw8_sanity_3():
    """Third sanity test for the Conv2dGw8 kernel."""
    params = Conv2dGw8Params(n=1, c=64, h=8, w=16, padding=1, r=3, s=3, has_bias=False)
    config = Conv2dGw8Config(groups=8, block_p=8, block_n=1)
    run_kernel_test(conv2d_gw8_kernel_factory, params, configs=[config])


def test_kernel_conv2d_gw8_wgrad_sanity():
    """Sanity test for the Conv2dGw8 wgrad kernel."""
    params = Conv2dGw8Params(n=4, c=64, h=16, w=32, padding=1, r=3, s=3)
    run_grad_kernel_test(conv2d_gw8_wgrad_kernel_factory, params)


def test_kernel_conv2d_gw8_wgrad_range_base_loop_failure_case():
    """This test failed when using a range-based loop over kernel-rows r.

    It was fixed by reverting to a classical for loop.

    CUDA error: too many resources requested for launch
    """
    params = Conv2dGw8Params(
        n=64,
        c=1632,
        h=10,
        w=10,
        padding=(2, 2),
        r=5,
        s=5,
        has_bias=True,
        group_width=8,
        stride=1,
    )
    config = Conv2dGw8WgradConfig(
        groups=4, block_h=8, block_n_iters=1, warp_n=4, warp_s=1
    )
    run_grad_kernel_test(conv2d_gw8_wgrad_kernel_factory, params, configs=[config])


def test_functional_conv2d_gw8_grad_sanity():
    """Sanity test for the Conv2dGw8 functional gradient."""
    params = Conv2dGw8Params(
        n=4, c=128, h=32, w=32, padding=(2, 0), r=4, s=1, has_bias=True
    )
    run_grad_function_test(conv2d_gw8, params)


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_kernel_conv2d_gw8(params: Conv2dGw8Params):
    """Test the Conv2dGw8 kernel."""
    run_kernel_test(conv2d_gw8_kernel_factory, params)


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_kernel_conv2d_gw8_wgrad(params: Conv2dGw8Params):
    """Test the Conv2dGw8 wgrad kernel."""
    run_grad_kernel_test(conv2d_gw8_wgrad_kernel_factory, params)


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_kernel_conv2d_gw8_igrad(params: Conv2dGw8Params):
    """Test the Conv2dGw8 dgrad kernel."""
    run_grad_kernel_test(conv2d_gw8_kernel_factory, params, igrad=True)


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_functional_conv2d_gw8(params: Conv2dGw8Params):
    """Test the conv2d_gw8 op."""
    run_function_test(conv2d_gw8, params)


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_functional_conv2d_gw8_grad(params: Conv2dGw8Params):
    """Test the gradients of the conv2d_gw8 op."""
    run_grad_function_test(conv2d_gw8, params)


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_conv2d_gw8_layer(params: Conv2dGw8Params):
    """Test the Conv2dGw8 layer."""
    run_layer_test(Conv2dGw8, params)


@pytest.mark.parametrize("params", _random_sample_test_params())
def test_conv2d_gw8_layer_torchcompile(params: Conv2dGw8Params):
    """Test the Conv2dGw8 layer with torchcompile."""
    run_layer_test(Conv2dGw8, params, torchcompile=True)


def test_conv2d_gw8_op_check():
    """Test whether  the conv2d_gw8 custom op has been registered with PyTorch correctly."""
    params = Conv2dGw8Params(n=4, c=64, h=16, w=32, padding=1, r=3, s=3, has_bias=True)
    run_opcheck_test(conv2d_gw8, params)
