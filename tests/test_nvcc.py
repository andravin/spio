"""Test for NVCC compiler path detection."""

import pytest

import spio.compiler


@pytest.mark.skip(reason="NVCC support not requried by default.")
def test_nvcc():
    """Test NVCC compiler path is found."""
    assert spio.compiler.nvcc_full_path() is not None
