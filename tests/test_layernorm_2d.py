"""Unit tests for the LayerNorm2d kernels, function, and layers."""

import pytest

from spio.src_tests import run_kernel_test, run_function_test
from spio.kernels import layernorm_2d_kernel_factory, LayerNorm2dParams
from spio.functional import layernorm_2d


_channels_non_power_of_2_params = [
    LayerNorm2dParams(n=4, c=c, h=16, w=32, elementwise_affine=True, bias=True)
    for c in [8 * c8 for c8 in range(1, 16)]
]

_odd_sizes_params = [
    LayerNorm2dParams(n=3, c=256, h=37, w=23, elementwise_affine=True, bias=True),
    LayerNorm2dParams(n=103, c=256, h=101, w=107, elementwise_affine=True, bias=True),
]


def test_layernorm_2d_kernel_sanity():
    """Simple test of the layernorm_2d kernel."""
    params = LayerNorm2dParams(n=4, c=256, h=16, w=32)
    run_kernel_test(layernorm_2d_kernel_factory, params)


@pytest.mark.parametrize("params", _odd_sizes_params)
def test_layernorm_2d_kernel_odd_sizes(params: LayerNorm2dParams):
    """Test layernorm2d kernel with off feature map and batch sizes."""
    run_kernel_test(layernorm_2d_kernel_factory, params)


@pytest.mark.parametrize("params", _channels_non_power_of_2_params)
def test_layernorm_2d_kernel_channels_not_power_of_2(params: LayerNorm2dParams):
    """Test layernorm_2d kernel with number of channels not equal to a power of 2."""
    run_kernel_test(layernorm_2d_kernel_factory, params)


def test_layernorm_2d_kernel_wide():
    """Simple test of the layernorm_2d kernel."""
    params = LayerNorm2dParams(n=4, c=2048, h=16, w=32)
    run_kernel_test(layernorm_2d_kernel_factory, params)


def test_layernorm_2d_kernel_too_wide():
    """Simple test of the layernorm_2d kernel."""
    with pytest.raises(AssertionError):
        params = LayerNorm2dParams(n=4, c=2049, h=16, w=32)
        run_kernel_test(layernorm_2d_kernel_factory, params)


def test_functional_layernorm_sanity():
    """Simple test of the layernorm_2d function."""
    params = LayerNorm2dParams(
        n=4, c=256, h=16, w=32, elementwise_affine=False, bias=False
    )
    run_function_test(layernorm_2d, params)
