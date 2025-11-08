"""Code generator for tensor-matrix multiplication operations."""

from typing import Set
from dataclasses import dataclass

from .tensor import Tensor
from .dim import dim_name_to_dim_or_fold_class_name


@dataclass
class Matmul:
    """Generates matrix multiplication code for tensor operands.

    Performs the operation:
        d = a x b + c

    This code implements a generalized tensor-matrix that supports:
    - Any number of independent dimensions.
    - Any number of reduction dimensions.

    An independent dimension is one that appears in either operand A or B but not both.
    A reduction dimension appears in both operands A and B.
    C and D must have all the independent dimensions and none of the reduction dimensions.

    """

    tensor_a: Tensor
    tensor_b: Tensor
    tensor_c: Tensor
    tensor_d: Tensor
    function_name: str = "tensor_matmul"
    use_zigzag: bool = True  # Add zigzag parameter

    def generate(self) -> str:
        """Generate fully unrolled matrix multiplication code with zigzag for all dimensions."""
        # Identify dimension categories
        a_dims = set(self.tensor_a.dim_names)
        b_dims = set(self.tensor_b.dim_names)

        reduction_dims = sorted(list(a_dims.intersection(b_dims)))
        a_only_dims = sorted(list(a_dims - set(reduction_dims)))
        b_only_dims = sorted(list(b_dims - set(reduction_dims)))

        # Get dimension sizes
        dim_sizes = {}
        for dim in a_dims:
            dim_sizes[dim] = self.tensor_a.dims[dim]
        for dim in b_dims:
            dim_sizes[dim] = self.tensor_b.dims[dim]

        # Generate code
        lines = []

        # Generate function signature
        lines.append("/**")
        lines.append(
            f" * Optimized matrix multiplication function for specific tensor formats."
        )
        lines.append(f" * - Tensor A dimensions: {', '.join(a_dims)}")
        lines.append(f" * - Tensor B dimensions: {', '.join(b_dims)}")
        lines.append(f" * - Output dimensions: {', '.join(a_only_dims + b_only_dims)}")
        lines.append(f" * - Reduction dimensions: {', '.join(reduction_dims)}")
        lines.append(f" * Using reduction-first traversal order.")
        if self.use_zigzag:
            lines.append(f" * Using zigzag traversal for all dimensions.")
        lines.append(
            " * Note: Tensors must be initialized before calling this function."
        )
        lines.append(" */")

        # Function declaration with concrete types
        lines.append("DEVICE void")
        lines.append(f"{self.function_name}(")
        lines.append(f"    const {self.tensor_a.class_name}& a,")
        lines.append(f"    const {self.tensor_b.class_name}& b,")
        lines.append(f"    const {self.tensor_c.class_name}& c,")
        lines.append(f"    {self.tensor_d.class_name}& d")
        lines.append(") {")

        # Function body
        indent = "    "

        # Generate all index combinations
        all_indices = []

        # Determine traversal order
        traversal_order = reduction_dims + a_only_dims + b_only_dims

        # Generate cartesian product of all indices
        index_values = {}
        for dim in traversal_order:
            size = dim_sizes[dim]
            index_values[dim] = list(range(size))

        # Build all index combinations
        def build_indices(dimensions, current_indices=None, depth=0):
            if current_indices is None:
                current_indices = {}

            if depth == len(dimensions):
                all_indices.append(current_indices.copy())
                return

            dim = dimensions[depth]

            # Implement zigzag pattern for ALL dimensions after the first one
            values = index_values[dim]

            # Zigzag applies to any dimension after the first one
            should_zigzag = self.use_zigzag and depth > 0

            # Alternate direction based on the parity of preceding indices
            if should_zigzag:
                # Sum of all previous indices to determine direction
                indices_sum = sum(current_indices.get(d, 0) for d in dimensions[:depth])

                # Reverse the traversal order if the sum is odd
                if indices_sum % 2 == 1:
                    values = values[::-1]

            # Process all values for this dimension
            for val in values:
                current_indices[dim] = val
                build_indices(dimensions, current_indices, depth + 1)

        # Generate all index combinations
        build_indices(traversal_order)

        # Generate unrolled matrix multiply operations
        lines.append(f"{indent}// Fully unrolled matrix multiply operations")

        for idx in all_indices:
            # Build indexed tensor references
            a_ref = "*a"
            b_ref = "*b"
            c_ref = "*c"
            d_ref = "*d"

            for dim in sorted(a_dims):
                a_ref += f"[{dim_name_to_dim_or_fold_class_name(dim)}({idx[dim]})]"

            for dim in sorted(b_dims):
                b_ref += f"[{dim_name_to_dim_or_fold_class_name(dim)}({idx[dim]})]"

            # Output and accumulator tensors have only independent dimensions
            for dim in sorted(a_only_dims + b_only_dims):
                c_ref += f"[{dim_name_to_dim_or_fold_class_name(dim)}({idx[dim]})]"
                d_ref += f"[{dim_name_to_dim_or_fold_class_name(dim)}({idx[dim]})]"

            # Determine zigzag parity for B dimensions
            is_zag = False
            if self.use_zigzag and b_only_dims:
                # Find indices of all dimensions before the first B-only dimension
                first_b_dim = b_only_dims[0]
                first_b_dim_pos = traversal_order.index(first_b_dim)
                indices_before_b = [
                    idx.get(dim, 0)
                    for dim in traversal_order[:first_b_dim_pos]
                    if dim in idx
                ]
                is_zag = sum(indices_before_b) % 2 == 1

            # Use reverse version on "zag" iterations for all fragment types
            if is_zag:
                lines.append(
                    f"{indent}mma_trans_reverse({d_ref}, {a_ref}, {b_ref}, {c_ref});"
                )
            else:
                lines.append(f"{indent}mma_trans({d_ref}, {a_ref}, {b_ref}, {c_ref});")

        # Close function
        lines.append("}")

        code = "\n".join(lines)
        return code

    def _build_subscript_chain(self, dims: Set[str], available_dims: Set[str]) -> str:
        """Build a chain of subscript operators for the given dimensions."""
        subscripts = ""
        for dim in sorted(list(dims.intersection(available_dims))):
            subscripts += f"[{dim}_idx]"
        return subscripts

    def _get_dim_class_name(self, dim_name: str) -> str:
        """Get the C++ class name for a dimension."""
        return dim_name_to_dim_or_fold_class_name(dim_name)


def generate_tensor_matmul(
    tensor_a: Tensor,
    tensor_b: Tensor,
    tensor_c: Tensor,
    tensor_d: Tensor,
    function_name: str = "tensor_matmul",
    use_zigzag: bool = True,
) -> str:
    """Generate optimized code for matrix multiplication of tensors."""
    generator = Matmul(
        tensor_a=tensor_a,
        tensor_b=tensor_b,
        tensor_c=tensor_c,
        tensor_d=tensor_d,
        function_name=function_name,
        use_zigzag=use_zigzag,
    )
    return generator.generate()
