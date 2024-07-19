def generate(
    gen_specs,
    header_file_name,
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
    with open(header_file_name, "w") as file:
        file.write(code)


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
