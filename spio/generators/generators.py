"""Generate CUDA code using generator specifications."""

from itertools import count
from typing import Iterable, List

from .dim import Dim, _get_dim_name_and_stride, BUILTIN_DIM_NAMES
from .fold import Fold
from .tensor import Tensor
from .compound_index import CompoundIndex
from .fragment import FragmentBase
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

    # 0. Recursively collect all used generators (using dict keyed by id to handle unhashable specs)
    # These are the Dim/Fold objects referenced by Tensors etc. that need code generated.
    def collect_recursive(spec, collected):
        """Recursively collect all used generators from a spec."""
        spec_id = id(spec)
        if spec_id in collected:
            return
        collected[spec_id] = spec
        if hasattr(spec, "used_generators"):
            for used_spec in spec.used_generators():
                collect_recursive(used_spec, collected)

    used_generators_by_id = {}
    for spec in gen_specs:
        collect_recursive(spec, used_generators_by_id)

    # Auto-name any that don't have names yet
    for used_spec in used_generators_by_id.values():
        if (
            hasattr(used_spec, "get_class_name")
            and hasattr(used_spec, "_set_class_name")
            and used_spec.get_class_name() is None
        ):
            _generate_name(used_spec)

    # Also auto-name any top-level specs that don't have names
    for spec in gen_specs:
        if (
            hasattr(spec, "get_class_name")
            and hasattr(spec, "_set_class_name")
            and spec.get_class_name() is None
        ):
            _generate_name(spec)

    # Combine all generator specifications - include ALL used generators
    # (they may have been auto-named in Dims or here, but still need code generation)
    gen_specs = list(gen_specs) + list(used_generators_by_id.values())

    # 1. Find explicitly declared Fold specs
    explicit_folds = {spec.fold_name: spec for spec in gen_specs if isinstance(spec, Fold)}

    # Track all fold names - these are NOT base dimensions
    fold_names = set(explicit_folds.keys())

    # Track all dimension names used as dim_name in fold specs (the base dim of each fold)
    folded_dim_names = {spec.dim_name for spec in gen_specs if isinstance(spec, Fold)}

    # 2. Find all dimension names used in any specs
    all_dim_names = set()
    for spec in gen_specs:
        if hasattr(spec, "dim_names"):
            all_dim_names.update(spec.dim_names)

    # 3. Extract base dimensions and implicit fold dimensions
    base_dims = set()
    implicit_folds = {}  # Use dict keyed by fold_name to avoid duplicates

    for name in all_dim_names:
        # Skip names that are known fold names (explicit folds from Fold objects)
        if name in fold_names:
            continue

        base_name, stride = _get_dim_name_and_stride(name)
        if stride is not None:
            # This is an implicit fold dimension (e.g., "K16" when K16 wasn't explicitly declared)
            if name not in folded_dim_names and name not in explicit_folds:
                # Only create implicit folds if not explicitly declared
                # and not used as dim_name in a fold spec
                implicit_folds[name] = Fold(base_name, stride, fold_name=name)
                fold_names.add(name)  # Add to fold_names to exclude from base dims
            base_dims.add(Dim(base_name))
        else:
            # No numeric suffix - this is a base dimension name
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
        if isinstance(spec, FragmentBase):
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
        if isinstance(spec, FragmentBase):
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


def _counter_to_alpha(n: int) -> str:
    """Convert a counter to an alphabetic suffix (a, b, c, ..., aa, ab, ...)."""
    result = []
    while n > 0:
        n -= 1  # Make 1-indexed (1->a, 2->b, etc.)
        result.append(chr(ord("a") + (n % 26)))
        n //= 26
    return "".join(reversed(result)) if result else "a"


def _generate_name(spec: GenSpecs) -> None:
    """Generate a class name for an unnamed generator specification.

    Note: We use alphabetic suffixes (a, b, c, ...) to avoid numeric suffixes
    like _Dim_1 which would be misinterpreted as fold dimensions.
    """
    prefix = type(spec).__name__
    counter = next(_anon_counter)
    alpha_suffix = _counter_to_alpha(counter)
    spec._set_class_name(f"_Anon{prefix}{alpha_suffix}")
