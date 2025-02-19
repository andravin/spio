"""
Run all C++ unit tests as a single pytest test.

The C++ tests conver generated index and tensor classes. These classes
work in both C++ and CUDA programs.
"""

from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile
import os

from importlib_resources import files as importlib_resources_files
import pytest

from spio.generators import (
    TensorSpec,
    IndexSpec,
    CheckerboardIndexSpec,
    FragmentIndexSpec,
    FragmentLoadIndexSpec,
)
import spio.compiler


CPP_SOURCES = [
    "test_index.cpp",
    "test_tensor.cpp",
    "test_fragment_index.cpp",
    "test_mathutil.cpp",
    "test_strip_loader_params.cpp",
]

TRUTHS = ["true", "1", "yes", "y", "t"]

ENABLE_CPP_TESTS = os.environ.get("SPIO_ENABLE_CPP_TESTS", "false").lower() in TRUTHS


def compile_cpp_tests(extra_cpp_test_files=None):
    """Compile C++ tests with NVCC."""
    if extra_cpp_test_files is None:
        extra_cpp_test_files = []
    includes = [
        importlib_resources_files("spio.include"),
        importlib_resources_files("spio.src_tests"),
    ]
    sources = [
        importlib_resources_files("spio.src_tests") / src for src in CPP_SOURCES
    ] + extra_cpp_test_files
    includes = [str(include) for include in includes]
    return spio.compiler.compile_with_nvcc(sources=sources, includes=includes, run=True)


@pytest.mark.skipif(
    not ENABLE_CPP_TESTS, reason="NVCC support not requried by default."
)
def test_cpp_tests():
    """Run all C++ unit tests."""
    headers = [
        '#include "utest.h"',
        spio.generators.index.index_header(),
        spio.generators.tensor._tensor_header(),
        spio.generators.checkerboard._checkerboard_header(),
        spio.generators.fragment_index._fragment_index_header(),
        spio.generators.fragment_index._fragment_load_index_header(),
    ]
    sources = [
        _test_generate_index(),
        _test_generate_dense_tensor(),
        _test_generate_checkerboard_index(),
        _test_generate_tensor_with_strides(),
        _test_checkerboard_tensor(),
        _test_fragment_index(),
        _test_fragment_load_index(),
    ]
    code = "\n".join(headers + sources)
    test_source_file = NamedTemporaryFile(prefix="spio_", suffix=".cpp")
    with open(test_source_file.name, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        compile_cpp_tests([test_source_file.name])
    except CalledProcessError as e:
        assert False, f"{e.stdout} {e.stderr}"


def _test_generate_index():
    """Return the C++ source code that tests a custom index class."""

    my_index_code = IndexSpec(
        "MyIndex", {"n": 4, "h": 32, "w": 64, "c": 128}
    ).generate()
    size = 4 * 32 * 64 * 128
    test_code = f"""
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


def _test_generate_checkerboard_index():
    my_fused_index_code = IndexSpec(
        "MyCheckerboard", dict(c16=8, checkers=CheckerboardIndexSpec(r=16, cm2=2))
    ).generate()

    code = f"""
{my_fused_index_code}

UTEST(IndexSpec, checkerboard_fused_dim)
{{
    EXPECT_EQ(MyCheckerboard::size, 8 * 16 * 2);
    EXPECT_EQ(MyCheckerboard::C16, 8);
    EXPECT_EQ(MyCheckerboard::CHECKERS, 16 * 2);
    for (int offset = 0; offset < MyCheckerboard::size; ++offset) {{
        MyCheckerboard idx(offset);
        EXPECT_EQ(idx.c16(), offset / (16 * 2));
        int ckbd_offset = offset % 32;
        int ckbd_row = ckbd_offset / 8;
        EXPECT_EQ(idx.checkers().r(), ckbd_offset / 2);
        EXPECT_EQ(idx.checkers().cm2(), (ckbd_offset & 1) ^ (ckbd_row & 1));
    }}
}}
"""
    return code


def _test_generate_dense_tensor():
    """Return the C++ source code that tests a custom tensor class."""
    n = 7
    h = 16
    w = 33
    c = 42

    dense_tensor_code = TensorSpec(
        "DenseTensor", "const float", {"n": n, "h": h, "w": w, "c": c}
    ).generate()
    test_code = f"""

{dense_tensor_code}

UTEST(DenseTensor, offset_from_tensor)
{{
    constexpr int N = {n};
    constexpr int H = {h};
    constexpr int W = {w};
    constexpr int C = {c};
    constexpr int size = N * H * W *C;
    constexpr size_t num_bytes = sizeof(float) * size;

    float data[N * H * W * C];
    for (int n = 0; n < N; ++n) {{
        for (int h = 0; h < H; ++h) {{
            for (int w = 0; w < W; ++w) {{
                for (int c = 0; c < C; ++c) {{
                    data[n*(H*W*C) + h*(W*C) + w*C +c] = n*(H*W*C) + h*(W*C) + w*C + c;
                }}
            }}
        }}
    }}
    for (int n = 0; n < N; ++n) {{
        for (int h = 0; h < H; ++h) {{
            for (int w = 0; w < W; ++w) {{
                for (int c = 0; c < C; ++c) {{
                    EXPECT_EQ(*DenseTensor(data).n(n).h(h).w(w).c(c), n*(H*W*C) + h*(W*C) + w*C + c);
                }}
            }}
        }}
    }}
    EXPECT_EQ(DenseTensor::size, size);
    EXPECT_EQ(DenseTensor::num_bytes, static_cast<int>(num_bytes));
}}
"""
    return test_code


def _test_generate_tensor_with_strides():
    """Return the C++ source code that tests a custom tensor class."""
    n = 7
    h = 16
    w = 33
    c = 42

    stride_w = c + 2
    stride_h = (w + 1) * stride_w

    dense_tensor_code = TensorSpec(
        "StrideTensor",
        "const float",
        {"n": n, "h": h, "w": w, "c": c},
        strides={"h": stride_h, "w": stride_w},
    ).generate()
    test_code = f"""

{dense_tensor_code}

UTEST(StrideTensor, offset_from_tensor)
{{
    constexpr int N = {n};
    constexpr int H = {h};
    constexpr int W = {w};
    constexpr int C = {c};

    constexpr int stride_w = {stride_w};
    constexpr int stride_h = {stride_h};
    constexpr int stride_n = H * stride_h;
    constexpr int size = N * stride_n;
    constexpr size_t num_bytes = sizeof(float) * size;

    float data[N * stride_n];
    for (int n = 0; n < N; ++n) {{
        for (int h = 0; h < H; ++h) {{
            for (int w = 0; w < W; ++w) {{
                for (int c = 0; c < C; ++c) {{
                    data[n*stride_n + h*stride_h + w*stride_w +c] = n*(H*W*C) + h*(W*C) + w*C + c;
                }}
            }}
        }}
    }}
    for (int n = 0; n < N; ++n) {{
        for (int h = 0; h < H; ++h) {{
            for (int w = 0; w < W; ++w) {{
                for (int c = 0; c < C; ++c) {{
                    EXPECT_EQ(*StrideTensor(data).n(n).h(h).w(w).c(c), n*(H*W*C) + h*(W*C) + w*C + c);
                }}
            }}
        }}
    }}
    EXPECT_EQ(StrideTensor::size, size);
    EXPECT_EQ(StrideTensor::num_bytes, static_cast<int>(num_bytes));
}}
"""
    return test_code


def _test_checkerboard_tensor():
    c16 = 8
    r = 16
    cm2 = 2

    tensor_code = TensorSpec(
        "CheckerboardTensor",
        "const float",
        dict(c16=c16, checkers=CheckerboardIndexSpec(r=16, cm2=cm2)),
    ).generate()
    test_code = f"""

{tensor_code}

UTEST(CheckerboardTensor, offset_from_tensor)
{{
    constexpr int C16 = {c16};
    constexpr int R = {r};
    constexpr int CM2 = {cm2};
    constexpr int size = C16 * R * CM2;
    constexpr size_t num_bytes = sizeof(float) * size;

    float data[C16 * R * CM2];
    for (int c16 = 0; c16 < C16; ++c16) {{
        for (int r = 0; r < R; ++r) {{
            for (int cm2 = 0; cm2 < CM2; ++cm2) {{
                data[c16 * R * CM2 + r * CM2 + cm2] = c16 * R * CM2 + r * CM2 + cm2;
            }}
        }}
    }}

    for (int c16 = 0; c16 < C16; ++c16) {{
        for (int r = 0; r < R; ++r) {{
            for (int cm2 = 0; cm2 < CM2; ++cm2) {{
                EXPECT_EQ(
                    *CheckerboardTensor(data).c16(c16).checkers(r, cm2),
                    data[c16 * R * CM2 + spio::CheckerboardIndex<8>::offset(r, cm2)]
            );
            }}
        }}
    }}
}}
    """
    return test_code


def _test_fragment_index():
    index_a_code = FragmentIndexSpec("A", "MMA_M16_K16_F16_A", "r", "s").generate()
    index_b_code = FragmentIndexSpec("B", "MMA_N16_K16_F16_B", "s", "t").generate()
    index_c_code = FragmentIndexSpec("C", "MMA_M16_N16_F32_C", "r", "s").generate()
    test_code = f"""
{index_c_code}

UTEST(FragmentIndex, MMA_M16_K16_F16_A)
{{
    {index_a_code}

    for (int lane = 0; lane < 32; ++lane) {{
        for (int idx = 0; idx < A::size(); ++idx) {{
            EXPECT_EQ(A(lane).r(idx), spio::MMA_A_88_F16_Index(lane).i(idx));
            EXPECT_EQ(A(lane).s2(idx), spio::MMA_A_88_F16_Index(lane).k2(idx));
            EXPECT_EQ(A(lane).s8(idx), spio::MMA_A_88_F16_Index(lane).k8(idx));
            EXPECT_EQ(A(lane).s2m4(), spio::MMA_A_88_F16_Index(lane).k2m4());
        }}
    }}
}}

UTEST(FragmentIndex, MMA_N16_K16_F16_B)
{{
    {index_b_code}

    for (int lane = 0; lane < 32; ++lane) {{
        for (int idx = 0; idx < B::size(); ++idx) {{
            EXPECT_EQ(B(lane).t(idx), spio::MMA_B_88_F16_Index(lane).j(idx));
            EXPECT_EQ(B(lane).s2(idx), spio::MMA_B_88_F16_Index(lane).k2(idx));
            EXPECT_EQ(B(lane).s8(idx), spio::MMA_B_88_F16_Index(lane).k8(idx));
            EXPECT_EQ(B(lane).s2m4(), spio::MMA_B_88_F16_Index(lane).k2m4());
        }}
    }}
}}

UTEST(FragmentIndex, MMA_M16_N16_F32_C)
{{
    {index_c_code}

    for (int lane = 0; lane < 32; ++lane) {{
        for (int idx = 0; idx < C::size(); ++idx) {{
            EXPECT_EQ(C(lane).r(idx), spio::MMA_C_88_F32_Index(lane).i(idx));
            EXPECT_EQ(C(lane).s2(idx), spio::MMA_C_88_F32_Index(lane).j2(idx));
            EXPECT_EQ(C(lane).s8(idx), spio::MMA_C_88_F32_Index(lane).j8(idx));
            EXPECT_EQ(C(lane).s2m4(), spio::MMA_C_88_F32_Index(lane).j2m4());
        }}
    }}
}}
"""
    return test_code


def _test_fragment_load_index():
    index_a_code = FragmentLoadIndexSpec("A", "MMA_M16_K16_F16_A", "r", "s").generate()
    test_code = f"""
UTEST(FragmentLoadIndex, MMA_M16_K16_F16_A)
{{
    {index_a_code}

    for (int lane = 0; lane < 32; ++lane) {{
            EXPECT_EQ(A(lane).r(), spio::MMA_A_M16_K16_F16_LoadIndex(lane).i());
            EXPECT_EQ(A(lane).s8(), spio::MMA_A_M16_K16_F16_LoadIndex(lane).k8());
    }}
}}
"""
    return test_code
