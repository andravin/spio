"""Generate CUDA code using generator specifications."""

from typing import List, Set, Union

from .gen_specs import GenSpecs
from .dim import Dim, _get_dim_name_and_stride
from .fold import Fold
from .fragment import Fragment


def generate(
    gen_specs: List[GenSpecs],
    namespace: str = None,
) -> str:
    """Generate CUDA code from generator specifications.

    Automatically detects all dimensions used in the generator specifications
    and generates the corresponding custom dimension classes.

    Args:
        gen_specs: List of generator specifications.
        namespace: Optional namespace for the generated code.

    Returns:
        Generated CUDA code as a string.
    """
    declared_fold_specs = [spec for spec in gen_specs if isinstance(spec, Fold)]
    declared_fold_spec_names = [spec.fold_name for spec in declared_fold_specs]
    used_fold_or_dim_specs = _get_all_used_fold_or_dim_specs(
        gen_specs, declared_fold_spec_names
    )
    used_fold_specs = [
        spec for spec in used_fold_or_dim_specs if isinstance(spec, Fold)
    ]
    used_dim_specs = [spec for spec in used_fold_or_dim_specs if isinstance(spec, Dim)]

    user_data_types = _get_user_defined_data_types(gen_specs)
    code = _include_files()
    code += "\n"
    if namespace is not None:
        code += _start_namespace(namespace)
    for dim_spec in used_dim_specs:
        code += dim_spec.generate()
    for fold_spec in used_fold_specs:
        code += fold_spec.generate()
    for spec in gen_specs:
        if isinstance(spec, Dim):
            continue
        if isinstance(spec, Fold) and spec in used_fold_specs:
            continue
        if hasattr(spec, "generate_with_context"):
            code += spec.generate_with_context(user_data_types=user_data_types)
        else:
            code += spec.generate()
        code += "\n"
    if namespace is not None:
        code += _end_namespace()
    return code


def _get_user_defined_data_types(gen_specs: List[GenSpecs]) -> List[str]:
    """Get the names of all fragments in the generator specifications.

    Fragments can be used as a tensor data-type.
    """
    type_names = []
    for spec in gen_specs:
        if isinstance(spec, Fragment):
            type_names.append(spec.class_name)
    return type_names


def _get_all_used_fold_or_dim_specs(
    gen_specs: List[GenSpecs], declared_fold_spec_names: List[str]
) -> Set[str]:
    """Get the names of all dimensions in the generator specifications."""
    return set.union(
        *[
            _get_used_fold_or_dim_specs(spec, declared_fold_spec_names)
            for spec in gen_specs
        ]
    )


def _get_used_fold_or_dim_specs(
    spec: GenSpecs, declared_fold_spec_names: List[str]
) -> Set[Union[Fold, Dim]]:
    return set(
        _get_fold_or_dim_spec_from_name(name)
        for name in getattr(spec, "dim_names", [])
        if name not in declared_fold_spec_names
    )


def _get_fold_or_dim_spec_from_name(name: str) -> Union[Fold, Dim]:
    dim_name, stride = _get_dim_name_and_stride(name)
    if stride is not None:
        return Fold(name, dim_name, stride)
    else:
        return Dim(dim_name)


def _get_foldspec_names(gen_specs: List[GenSpecs]) -> Set[str]:
    return set(spec.fold_name for spec in gen_specs if isinstance(spec, Fold))


def _include_files():
    return """
#include "spio/allocator.h"
#include "spio/index_variadic.h"
#include "spio/tensor_variadic.h"
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
