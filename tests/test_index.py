"""
Run all C++ unit tests as a single pytest test.

The C++ tests
"""

from pathlib import Path
from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile

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


def compile_cpp_tests(extra_cpp_test_files=[]):
    src_dir = spio_cpp_tests_src_path()
    includes = [spio_include_path(), spio_cpp_tests_src_path()]
    sources = [str(src_dir / src) for src in CPP_SOURCES] + extra_cpp_test_files
    includes = [str(include) for include in includes]
    return spio.compile(sources=sources, includes=includes, run=True)


def test_cpp_tests():
    """Run all C++ unit tests."""

    test_generate_index_code = _test_generate_index()
    test_source_file = NamedTemporaryFile(prefix="spio_", suffix=".cpp")
    with open(test_source_file.name, 'w') as f:
        f.write(test_generate_index_code)

    try:
        compile_cpp_tests([test_source_file.name])
    except CalledProcessError as e:
        assert False, f"{e.stdout} {e.stderr}"


def _test_generate_index():
    """Return the C++ source code that tests a custom index class."""
    my_index_code = spio.generate_index("MyIndex", dict(n=4, h=32, w=64, c=128))
    header = spio.index_header()
    test_code = f"""
#include "utest.h"
{header}

using namespace spio;

{my_index_code}

UTEST(MyIndex, offset_from_index)
{{
    EXPECT_EQ(static_cast<int>(MyIndex().n(7)), 7 * (32 * 64 * 128));
    EXPECT_EQ(static_cast<int>(MyIndex().h(16)), 16 * (64 * 128));
    EXPECT_EQ(static_cast<int>(MyIndex().w(33)), 33 * 128);
    EXPECT_EQ(static_cast<int>(MyIndex().c(42)), 42);
}}

UTEST(MyIndex, index_from_offset)
{{
    int offset = 532523;
    MyIndex idx(offset);
    EXPECT_EQ(idx.n(), offset / (32 * 64 * 128));
    EXPECT_EQ(idx.h(), (offset / (64 * 128)) % 32);
    EXPECT_EQ(idx.w(), (offset / 128) % 64);
    EXPECT_EQ(idx.c(), offset % 128);
}}
"""
    return test_code