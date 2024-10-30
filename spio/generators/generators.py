"""Generate CUDA code using generator specifications."""

from typing import List

from .gen_specs import GenSpecs


def generate(
    gen_specs: List[GenSpecs],
    namespace: str = None,
) -> str:
    """Generate CUDA code from generator specifications.

    Args:
        gen_specs: List of generator specifications.
        namespace: Optional namespace for the generated code.

    Returns:
        Generated CUDA code as a string.
    """
    code = _include_files()
    code += "\n"
    if namespace is not None:
        code += _start_namespace(namespace)
    for spec in gen_specs:
        code += spec.generate()
        code += "\n"
    if namespace is not None:
        code += _end_namespace()
    return code


def _include_files():
    return """
#include "spio/index.h"
#include "spio/tensor.h"
#include "spio/mma.cuh"
"""


def _start_namespace(namespace: str) -> str:
    return f"""

namespace {namespace} {{
"""


def _end_namespace() -> str:
    return """
}
"""
