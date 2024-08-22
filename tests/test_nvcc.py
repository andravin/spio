import spio


def test_nvcc():
    assert spio.nvcc_full_path() is not None
