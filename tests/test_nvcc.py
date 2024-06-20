import spio.compiler as compiler


def test_nvcc():
    assert compiler.nvcc_full_path() is not None
