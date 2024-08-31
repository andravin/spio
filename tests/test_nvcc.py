import spio.compiler

def test_nvcc():
    assert spio.compiler.nvcc_full_path() is not None
