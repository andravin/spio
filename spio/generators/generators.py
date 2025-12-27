"""Generate CUDA code using generator specifications."""

from itertools import count
from typing import Iterable, List

from .dim import Dim, _get_dim_name_and_stride, BUILTIN_DIM_NAMES
from .fold import Fold
from .tensor import Tensor
from .compound_index import CompoundIndex
from .fragment import Fragment
from .gen_specs import GenSpecs

# Counter for anonymous class names.
_anon_counter = count(1)


def generate(
    gen_specs: Iterable[GenSpecs],
    namespace: str = None,
    utest_dim_printers: bool = False,
) -> str:
    """Generate CUDA code from generator specifications.

    Args:
        gen_specs: Either a list of generator specifications or a Generators container.
        namespace: Optional C++ namespace to wrap the generated code.
        utest_dim_printers: Whether to generate dimension printers for unit testing.

    Returns:
        The generated CUDA code as a string.
    """

    # 0. Collect unnamed generators (using dict keyed by id to handle unhashable specs)
    used_generators_by_id = {}
    for spec in gen_specs:
        for used_spec in spec.used_generators():
            used_generators_by_id[id(used_spec)] = used_spec

    anon_genspecs = [
        used_spec
        for used_spec in used_generators_by_id.values()
        if used_spec.get_class_name() is None
    ]

    # .. and assign them class names.
    for spec in anon_genspecs:
        _generate_name(spec)

    # Combine all generator specifications.
    gen_specs = list(gen_specs) + anon_genspecs

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
    implicit_folds = {}  # Use dict keyed by fold_name to avoid duplicates

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
                implicit_folds[name] = Fold(base_name, stride, fold_name=name)
                fold_aliases.add(name)  # Add to fold_aliases to exclude from base dims
        base_dims.add(Dim(base_name))

    # 4. Make sure all base dimensions for folds are created
    for fold in list(explicit_folds.values()) + list(implicit_folds.values()):
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
            if dim.dim_name not in BUILTIN_DIM_NAMES:
                code += dim.generate()
        code += "\n"

    # Group 2: Fold aliases
    all_folds = sorted(
        list(explicit_folds.values()) + list(implicit_folds.values()),
        key=lambda x: x.fold_name,
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
        elif isinstance(spec, CompoundIndex):
            indices.append(spec)
        else:
            others.append(spec)

    # Generate fragments
    if fragments:
        code += "// Fragment types\n"
        for fragment in fragments:
            code += fragment.generate()
        code += "\n"

    # Track which specs have been generated to avoid duplicates
    # Pre-populate with Dims and Folds that were already generated above
    generated_specs = set()
    for dim in base_dims:
        generated_specs.add(id(dim))
    for fold in all_folds:
        generated_specs.add(id(fold))
    for fragment in fragments:
        generated_specs.add(id(fragment))

    def generate_spec(spec):
        """Generate a spec and its dependencies, avoiding duplicates."""
        nonlocal code
        if id(spec) in generated_specs:
            return
        # Skip Dims and Folds - they're handled separately above
        if isinstance(spec, (Dim, Fold)):
            return
        # First generate any dependencies (used_generators)
        for used_spec in spec.used_generators():
            generate_spec(used_spec)
        # Then generate this spec
        generated_specs.add(id(spec))
        if hasattr(spec, "generate_with_context"):
            code += spec.generate_with_context(user_data_types=user_data_types)
        else:
            code += spec.generate()
        code += "\n"

    # Generate tensors (with their dependencies)
    if tensors:
        code += "// Tensor types\n"
        for tensor in tensors:
            generate_spec(tensor)

    # Generate indices
    if indices:
        code += "// CompoundIndex types\n"
        for index in indices:
            generate_spec(index)

    # Generate other specs (that haven't been generated as dependencies)
    remaining_others = [s for s in others if id(s) not in generated_specs]
    if remaining_others:
        code += "// Other types\n"
    for spec in remaining_others:
        generate_spec(spec)

    if namespace is not None:
        code += _end_namespace()

    # Optionally dimension printers used by the utest.h unit testing framework.
    if utest_dim_printers and base_dims:
        code += "\n"
        code += "// Dim printers for utest.h unit testing framework.\n"
        for dim in sorted(base_dims, key=lambda x: x.dim_name):
            code += f"UTEST_DIM_PRINTER({dim.dim_name});\n"

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
#include "spio/typed_dims.h"
"""


def _start_namespace(namespace: str) -> str:
    return f"namespace {namespace} {{\n"


def _end_namespace() -> str:
    return "}\n"


def _generate_name(spec: GenSpecs) -> None:
    """Generate a class name for an unnamed generator specification."""
    prefix = type(spec).__name__
    spec._set_class_name(f"_{prefix}_{next(_anon_counter)}")
