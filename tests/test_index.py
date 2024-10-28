"""
Run all C++ unit tests as a single pytest test.

The C++ tests conver generated index and tensor classes. These classes
work in both C++ and CUDA programs.
"""

from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile

from importlib_resources import files as importlib_resources_files
import pytest

import spio.generators
import spio.compiler


CPP_SOURCES = ["test_index.cpp"]


def compile_cpp_tests(extra_cpp_test_files=None):
    """Compile C++ tests with NVCC."""
    if extra_cpp_test_files is None:
        extra_cpp_test_files = []
    includes = [
        importlib_resources_files("spio.include"),
        importlib_resources_files("spio.src_tests"),
    ]
    sources = [
        importlib_resources_files("spio_src_tests") / src for src in CPP_SOURCES
    ] + extra_cpp_test_files
    includes = [str(include) for include in includes]
    return spio.compiler.compile_with_nvcc(sources=sources, includes=includes, run=True)


@pytest.mark.skip(reason="NVCC support not requried by default.")
def test_cpp_tests():
    """Run all C++ unit tests."""
    test_source = _test_generate_index()
    test_source += _test_generate_tensor()
    test_source_file = NamedTemporaryFile(prefix="spio_", suffix=".cpp")
    with open(test_source_file.name, "w", encoding="utf-8") as f:
        f.write(test_source)
    try:
        compile_cpp_tests([test_source_file.name])
    except CalledProcessError as e:
        assert False, f"{e.stdout} {e.stderr}"


def _test_generate_index():
    """Return the C++ source code that tests a custom index class."""
    my_index_code = spio.generators.index._generate_index(
        "MyIndex", {"n": 4, "h": 32, "w": 64, "c": 128}
    )
    size = 4 * 32 * 64 * 128
    header = spio.generators.index.index_header()
    test_code = f"""
#include "utest.h"

{header}

{my_index_code}

UTEST(MyIndex, index_from_offset)
{{
    int offset = 532523;
    MyIndex idx(offset);
    EXPECT_EQ(idx.n(), offset / (32 * 64 * 128));
    EXPECT_EQ(idx.h(), (offset / (64 * 128)) % 32);
    EXPECT_EQ(idx.w(), (offset / 128) % 64);
    EXPECT_EQ(idx.c(), offset % 128);
}}

UTEST(MyIndex, size)
{{
    EXPECT_EQ(MyIndex::size, {size});
}}
"""
    return test_code


def _test_generate_tensor():
    """Return the C++ source code that tests a custom tensor class."""
    n = 7
    h = 16
    w = 33
    c = 42

    my_tensor_code = spio.generators.tensor._generate_tensor(
        "MyTensor", "const float", {"n": n, "h": h, "w": w, "c": c}
    )
    header = spio.generators.tensor._tensor_header()
    test_code = f"""

{header}

{my_tensor_code}

UTEST(MyTensor, offset_from_tensor)
{{
    constexpr int N = {n};
    constexpr int H = {h};
    constexpr int W = {w};
    constexpr int C = {c};

    float data[N * H * W * C];
    for (int n = 0; n < N; ++n) {{
        for (int h = 0; h < H; ++h) {{
            for (int w = 0; w < W; ++w) {{
                for (int c = 0; c < C; ++c) {{
                    data[n*(H*W*C) + h*(W*C) + w*C +c] = n*(H*W*C) + h*(W*C) + w*C +c;
                }}
            }}
        }}
    }}
    for (int n = 0; n < N; ++n) {{
        for (int h = 0; h < H; ++h) {{
            for (int w = 0; w < W; ++w) {{
                for (int c = 0; c < C; ++c) {{
                    EXPECT_EQ(*MyTensor(data).n(n).h(h).w(w).c(c), n*(H*W*C) + h*(W*C) + w*C +c);
                }}
            }}
        }}
    }}
    EXPECT_EQ(MyTensor::num_bytes, static_cast<int>(sizeof(float) * N * H * W *C));
}}
"""
    return test_code
