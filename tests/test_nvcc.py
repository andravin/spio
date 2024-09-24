import pytest

import spio.compiler

@pytest.mark.skip(reason="NVCC support not requried by default.")
def test_nvcc():
    assert spio.compiler.nvcc_full_path() is not None
