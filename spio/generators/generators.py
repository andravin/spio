"""Generate CUDA code using generator specifications."""

from typing import List, Set

from .gen_specs import GenSpecs
from .dim import DimSpec, _get_dim_name_and_stride
from .fold import FoldSpec


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
    fold_specs = _get_foldspec_names(gen_specs)
    dim_names = _get_unique_dim_names(gen_specs)
    dim_names.difference_update(fold_specs)
    dim_names = sorted(dim_names)
    for dim_name in dim_names:
        code += DimSpec(dim_name).generate()
    for spec in gen_specs:
        if isinstance(spec, DimSpec):
            continue
        code += spec.generate()
        code += "\n"
    if namespace is not None:
        code += _end_namespace()
    return code


def _get_unique_dim_names(gen_specs: List[GenSpecs]) -> Set[str]:
    """Get the names of all dimensions in the generator specifications."""
    dim_names = set()
    for spec in gen_specs:
        names = _get_spec_dim_names(spec)
        dim_names.update(names)
    return dim_names


def _get_spec_dim_names(spec: GenSpecs) -> Set[str]:
    if hasattr(spec, "dim_names"):
        return set(_get_dim_name_and_stride(name)[0] for name in spec.dim_names)
    else:
        return set()


def _get_foldspec_names(gen_specs: List[GenSpecs]) -> Set[str]:
    return set(spec.fold_name for spec in gen_specs if isinstance(spec, FoldSpec))


def _include_files():
    return """
#include "spio/index.h"
#include "spio/tensor.h"
#include "spio/dim.h"
"""


def _start_namespace(namespace: str) -> str:
    return f"""

namespace {namespace} {{
"""


def _end_namespace() -> str:
    return """
}
"""
