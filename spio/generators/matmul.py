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
    reduction_first: bool = True

    def generate(self) -> str:
        """Generate optimized matrix multiplication code."""
        # Identify dimension categories
        a_dims = set(self.tensor_a.dim_names)
        b_dims = set(self.tensor_b.dim_names)

        reduction_dims = sorted(list(a_dims.intersection(b_dims)))
        a_only_dims = sorted(list(a_dims - set(reduction_dims)))
        b_only_dims = sorted(list(b_dims - set(reduction_dims)))

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
        if self.reduction_first:
            lines.append(f" * Using reduction-first traversal order.")
        else:
            lines.append(f" * Using independent-first traversal order.")
        lines.append(" * Note: Tensors must be initialized before calling this function.")
        lines.append(" */")

        # Function declaration with concrete types
        lines.append("DEVICE void")
        lines.append(f"{self.function_name}(")
        lines.append(f"    const {self.tensor_a.class_name}& a,")
        lines.append(f"    const {self.tensor_b.class_name}& b,")
        lines.append(f"    const {self.tensor_c.class_name}& c,")
        lines.append(f"    {self.tensor_d.class_name}& d")
        lines.append(") {")

        # Function body with indentation
        indent = "    "

        # Generate loop structure based on dimension ordering preference
        if self.reduction_first:
            # Process dimensions in this order: reduction, a_only, b_only
            all_dims_order = reduction_dims + a_only_dims + b_only_dims
        else:
            # Process dimensions in this order: a_only, b_only, reduction
            all_dims_order = a_only_dims + b_only_dims + reduction_dims

        # Generate nested loops
        processed_dims = set()

        for dim in all_dims_order:
            processed_dims.add(dim)

            # Get dimension info
            dim_size = None
            if dim in a_dims:
                dim_size = self.tensor_a.dims[dim]
            else:
                dim_size = self.tensor_b.dims[dim]

            dim_class = self._get_dim_class_name(dim)

            # Determine if this is a reduction dimension
            is_reduction = dim in reduction_dims
            dim_type = "Reduction" if is_reduction else "Independent"

            # Generate loop using range-based syntax
            lines.append(f"{indent}// {dim_type} dimension {dim}")
            lines.append(f"{indent}for (auto {dim}_idx : range({dim_class}({dim_size}))) {{")
            indent += "    "

        # Generate matrix multiply operation at innermost level with all indices
        lines.append(f"{indent}// Perform matrix multiply with consolidated indices")

        # Build subscript chain for each tensor
        a_subscripts = self._build_subscript_chain(a_dims, processed_dims)
        b_subscripts = self._build_subscript_chain(b_dims, processed_dims)

        # Use set operations for output dimensions
        output_dims = set(a_only_dims + b_only_dims)
        d_subscripts = self._build_subscript_chain(output_dims, processed_dims)
        c_subscripts = self._build_subscript_chain(output_dims, processed_dims)

        lines.append(f"{indent}mma_trans(")
        lines.append(f"{indent}    *d{d_subscripts},")
        lines.append(f"{indent}    *a{a_subscripts},")
        lines.append(f"{indent}    *b{b_subscripts},")
        lines.append(f"{indent}    *c{c_subscripts}")
        lines.append(f"{indent});")

        # Close all loops
        for _ in range(len(processed_dims)):
            indent = indent[:-4]
            lines.append(f"{indent}}}")

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
    reduction_first: bool = True,
) -> str:
    """Generate optimized code for matrix multiplication of tensors."""
    generator = Matmul(
        tensor_a=tensor_a,
        tensor_b=tensor_b,
        tensor_c=tensor_c,
        tensor_d=tensor_d,
        function_name=function_name,
        reduction_first=reduction_first,
    )
    return generator.generate()
