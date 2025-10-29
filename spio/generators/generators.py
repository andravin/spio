"""Generate CUDA code using generator specifications."""

from typing import List

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
    """Generate CUDA code from generator specifications."""
    # 1. Find explicitly declared Fold specs
    explicit_folds = {
        spec.fold_name: spec for spec in gen_specs if isinstance(spec, Fold)
    }

    # Track all dimension names used as dim_name in fold specs
    folded_dim_names = {spec.dim_name for spec in gen_specs if isinstance(spec, Fold)}

    # Also track the fold_name values from explicit folds (e.g., "block_p")
    fold_aliases = set(explicit_folds.keys())

    # 2. Find all dimension names used in any specs
    all_dim_names = set()
    for spec in gen_specs:
        if hasattr(spec, "dim_names"):
            all_dim_names.update(spec.dim_names)

    # 3. Extract base dimensions and implicit fold dimensions
    base_dims = set()
    implicit_folds = set()

    for name in all_dim_names:
        # Skip names that are fold_aliases (like "block_p") since they're not base dimensions
        if name in fold_aliases:
            continue

        base_name, stride = _get_dim_name_and_stride(name)
        if stride is not None:
            # This is a fold dimension (e.g., "c4")
            if name not in folded_dim_names and name not in explicit_folds:
                # Only create implicit folds if not explicitly declared
                # and not used as dim_name in a fold spec
                implicit_folds.add(Fold(name, base_name, stride))
                fold_aliases.add(name)  # Add to fold_aliases to exclude from base dims
        base_dims.add(Dim(base_name))

    # 4. Make sure all base dimensions for folds are created
    for fold in list(explicit_folds.values()) + list(implicit_folds):
        # Extract the base dimension name from the fold's dim_name
        base_name, _ = _get_dim_name_and_stride(fold.dim_name)
        base_dims.add(Dim(base_name))

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
    all_folds = sorted(
        list(explicit_folds.values()) + list(implicit_folds), key=lambda x: x.fold_name
    )
    if all_folds:
        code += "// Fold aliases\n"
        for fold in all_folds:
            code += fold.generate()
        code += "\n"

    # Group 3: Generate other types by category
    fragments = []
    tensors = []
    indices = []
    others = []

    for spec in gen_specs:
        if isinstance(spec, (Dim, Fold)):
            continue
        if isinstance(spec, Fragment):
            fragments.append(spec)
        elif isinstance(spec, Tensor):
            tensors.append(spec)
        elif isinstance(spec, Index):
            indices.append(spec)
        else:
            others.append(spec)

    # Generate fragments
    if fragments:
        code += "// Fragment types\n"
        for fragment in fragments:
            code += fragment.generate()
        code += "\n"

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
    if others:
        code += "// Other types\n"
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
#include "spio/index.h"
#include "spio/tensor.h"
#include "spio/dim.h"
"""


def _start_namespace(namespace: str) -> str:
    return f"namespace {namespace} {{\n"


def _end_namespace() -> str:
    return "}\n"
