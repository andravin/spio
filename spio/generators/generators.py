def generate(
    gen_specs,
    namespace: str = None,
) -> None:
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
    return f"""
}}
"""
