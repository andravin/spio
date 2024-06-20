"""
Run all C++ unit tests as a single pytest test.

The C++ tests
"""
from pathlib import Path
from subprocess import CalledProcessError

import spio

CPP_SOURCES = ["test_index.cpp"]

def spio_path():
    """Return the filesystem path of the spio installation."""
    return Path(spio.__file__).parent.parent


def spio_src_path():
    return spio_path() / "src"


def spio_include_path():
    return spio_path() / "include"


def spio_cpp_tests_src_path():
    return spio_src_path() / "tests"


def compile_cpp_tests():
    src_dir = spio_cpp_tests_src_path()
    includes = [spio_include_path()]
    sources = [str(src_dir / src) for src in CPP_SOURCES]
    includes = [str(include) for include in includes]

    spio.compile(sources=sources, includes=includes, run=True)


def test_cpp_tests():
    """Run all C++ unit tests."""
    try:
        compile_cpp_tests()
    except CalledProcessError as e:
        assert False, f"{e.stdout} {e.stderr}"
