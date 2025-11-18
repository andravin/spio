"""Code generator for matrix fragment with named dimensions."""

from dataclasses import dataclass
from typing import Tuple

from .fragment_type import FragmentType


@dataclass
class Fragment:
    """Fragment code generator.

    Example:

        Define a Fragment spec in your kernel factory's specs like this:
            Fragment("Acc", FragmentType.M16_N8_F32_C, "qn", "k2")

        Use the generated class in your CUDA kernel like this:
            # Get element coordinates for this thread.
            int lane = threadIdx.x % 32;
            Acc:Index acc_idx(lane);
            auto qn_val = acc_idx.get<QN>();
            auto k2_val = acc_idx.get<K2>();

            # Define an accumulator and initialize it to zero.
            Acc acc;
            acc.zero();

    Attributes:
        class_name: Name of the fragment class.
        fragment_type: Type of the fragment (see spio.include.spio / fragment.cuh)
        row: Name of the row dimension.
        col: Name of the column dimension.
    """

    class_name: str
    fragment_type: FragmentType
    row: str
    col: str

    def generate(self) -> str:
        """Generate the fragment class code as a type alias."""
        row_dim = self.row.upper()
        col_dim = self.col.upper()
        fragment_class = f"spio::{self.fragment_type.value}<{row_dim}, {col_dim}>"
        return f"using {self.class_name} = {fragment_class};\n"

    @property
    def dim_names(self) -> Tuple[str, str]:
        """Return the names of the dimensions."""
        return (self.row, self.col)


def header() -> str:
    """Return the header file for the fragment classes."""
    return """
#include "spio/fragment.cuh"
"""
