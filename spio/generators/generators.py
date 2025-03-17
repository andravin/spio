"""Generate CUDA code using generator specifications."""

from typing import List, Set, Union

from .gen_specs import GenSpecs
from .dim import Dim, _get_dim_name_and_stride
from .fold import Fold
from .tensor import Tensor
from .index import Index
from .fragment import Fragment


def generate(
    gen_specs: List[GenSpecs],
    namespace: str = None,
) -> str:
    """Generate CUDA code from generator specifications.

    Automatically detects all dimensions used in the generator specifications
    and generates the corresponding custom dimension classes and fold aliases.

    Args:
        gen_specs: List of generator specifications.
        namespace: Optional namespace for the generated code.

    Returns:
        Generated CUDA code as a string.
    """
    # 1. Find explicitly declared Fold specs
    explicit_folds = {
        spec.fold_name: spec for spec in gen_specs if isinstance(spec, Fold)
    }

    # 2. Find all dimension names used in any specs
    all_dim_names = set()
    for spec in gen_specs:
        if hasattr(spec, "dim_names"):
            all_dim_names.update(spec.dim_names)

    # 3. Extract base dimensions and implicit fold dimensions
    base_dims = set()
    implicit_folds = set()
    fold_names = set(
        explicit_folds.keys()
    )  # Track all fold names to exclude them from dims

    for name in all_dim_names:
        base_name, stride = _get_dim_name_and_stride(name)
        if stride is not None:
            # This is a fold dimension (e.g., "c4")
            if name not in explicit_folds:
                # Only create implicit folds if not explicitly declared
                implicit_folds.add(Fold(name, base_name, stride))
                fold_names.add(name)  # Add to fold names to exclude from dims
            # Always need the base dimension
            base_dims.add(Dim(base_name))
        else:
            # This is a base dimension (only if not a fold name)
            if name not in fold_names:
                base_dims.add(Dim(name))

    # 4. Make sure all base dimensions for folds are created
    for fold in list(explicit_folds.values()) + list(implicit_folds):
        base_dims.add(Dim(fold.dim_name))

    # 5. Generate code in a structured way
    user_data_types = _get_user_defined_data_types(gen_specs)
    code = _include_files() + "\n"

    if namespace is not None:
        code += _start_namespace(namespace)

    # Group 1: Dimension classes
    if base_dims:
        code += "// Dimension classes\n"
        for dim in sorted(base_dims, key=lambda x: x.dim_name):
            code += dim.generate()
        code += "\n"

    # Group 2: Fold aliases
    all_folds = sorted(list(explicit_folds.values()) + list(implicit_folds), key=lambda x: x.fold_name)
    if all_folds:
        code += "// Fold aliases\n"
        for fold in all_folds:
            code += fold.generate()
        code += "\n"

    # Group 3: Generate other types by category
    tensors = []
    indices = []
    others = []

    for spec in gen_specs:
        if isinstance(spec, (Dim, Fold)):
            continue
        if isinstance(spec, Tensor):
            tensors.append(spec)
        elif isinstance(spec, Index):
            indices.append(spec)
        else:
            others.append(spec)

    # Generate tensors
    if tensors:
        code += "// Tensor types\n"
        for tensor in tensors:
            code += tensor.generate_with_context(user_data_types=user_data_types)
        code += "\n"

    # Generate indices
    if indices:
        code += "// Index types\n"
        for index in indices:
            code += index.generate_with_context(user_data_types=user_data_types)
        code += "\n"
        
    # Generate other specs
    for spec in others:
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
