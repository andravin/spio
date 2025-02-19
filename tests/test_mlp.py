"""Unit tests for the Mlp kernels, function, and layers."""

import pytest

from spio.src_tests import run_kernel_test
from spio.kernels import mlp_kernel_factory, MlpParams


@pytest.mark.skip(reason="MLP not yet implemented.")
def test_mlp_kernel_sanity():
    """Simple test of the mlp kernel."""
    params = MlpParams(x=256, c=64, r=256, k=64, bias=False, activation="relu")
    run_kernel_test(mlp_kernel_factory, params)
