from pathlib import Path

import spio


def spio_path() -> Path:
    """Return the filesystem path of the spio installation."""
    return Path(spio.__file__).parent.parent


def spio_include_path() -> Path:
    return spio_path() / "include"


def spio_src_path() -> Path:
    return spio_path() / "src"


def spio_cpp_tests_src_path() -> Path:
    return spio_src_path() / "tests"


def spio_kernels_path() -> Path:
    return spio_path() / "kernels"


def spio_test_kernels_path() -> Path:
    return spio_path() / "tests"


def spio_cubins_path() -> Path:
    return spio_path() / "cubins"
