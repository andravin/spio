import code
from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile
import os
from typing import Callable
import re

from importlib_resources import files as importlib_resources_files
import pytest

import spio.compiler
from spio.generators import *
from spio.generators import GENERATORS
from spio.util import env_var_is_true

DIM_EXAMPLES = [
    "01_commutativity",
    "02_cursor_movement",
    "03_folding",
    "04_projection",
    "05_compound_index",
]

ENABLE_CPP_TESTS = env_var_is_true("SPIO_ENABLE_CPP_TESTS")

UTEST_HEADER = '#include "utest.h"'

DIM_EXAMPLE_MAIN = "tutorial_main.cpp"

SPIO_PATTERN = re.compile(r"/\*@spio\s*(.*?)\s*@spio\*/", re.DOTALL)

# Build EVAL_NAMESPACE dynamically from GENERATORS
EVAL_NAMESPACE = {"__builtins__": {}, "dtype": dtype, "Dims": Dims, "Strides": Strides}
for name in GENERATORS:
    EVAL_NAMESPACE[name] = globals()[name]

PRE_INCLUDES = ["dim_test_common.h"]

RESOURCE_PACKAGES = [
    "spio.include",
    "spio.src_tests",
    "spio.src_tests.tutorial",
]


@pytest.mark.skipif(not ENABLE_CPP_TESTS, reason="NVCC support not required by default")
def test_tutorial():
    """Test all dim examples."""
    for example in DIM_EXAMPLES:
        try:
            _compile_dim_example(example)
        except CalledProcessError as e:
            pytest.fail(f"Dim example {example} failed to compile or run: {e}")


def _compile_dim_example(example_name: str):
    """Compile the typed dimensions example with nvcc and run it.

    Args:
        example_name: The name of the example to compile and run.

    Returns:
        The return code from running the compiled example.
    """
    generated_headers = []
    try:
        cpp_source = _get_source(example_name)
        specs = _extract_specs(cpp_source)
        if specs:
            generated_code = generate(specs, utest_dim_printers=True)
            with NamedTemporaryFile(
                prefix="spio_", suffix=".h", mode="w", delete=False
            ) as f:
                f.write(generated_code)
                generated_headers.append(f.name)

        example_file_name = example_name + ".cpp"
        cpp_sources = [example_file_name, DIM_EXAMPLE_MAIN]
        sources = [
            importlib_resources_files("spio.src_tests.tutorial") / cpp_source
            for cpp_source in cpp_sources
        ]
        includes = [str(importlib_resources_files(pkg)) for pkg in RESOURCE_PACKAGES]
        pre_includes = PRE_INCLUDES + generated_headers
        return spio.compiler.compile_with_nvcc(
            sources=sources, includes=includes, pre_includes=pre_includes, run=True
        )
    finally:
        for generated_header in generated_headers:
            os.unlink(generated_header)


def _get_source(example_name: str) -> str:
    """Get the source code for a dim example."""
    return (
        importlib_resources_files("spio.src_tests.tutorial")
        .joinpath(example_name + ".cpp")
        .read_text()
    )


def _extract_specs(source: str) -> Generators:
    """Extract generator specifications from source code.

    Generator specs can appear anywhere in the source code in this format:

    /*@spio
    A = Tensor(dtype.float, Dims(i=16, k=32))
    B = Tensor(dtype.float, Dims(k=32, j=64))
    @spio*/
    """
    matches = SPIO_PATTERN.findall(source)
    g = Generators()
    for match in matches:
        namespace = dict(EVAL_NAMESPACE)
        exec(match, namespace)
        for name, value in namespace.items():
            if name not in EVAL_NAMESPACE:
                setattr(g, name, value)
    return g
