"""Code generator for matrix fragment with named dimensions."""

from dataclasses import dataclass

from .fragment_index import (
    FragmentLoadIndexSpec,
    FragmentIndexSpec,
    fragment_load_supported,
)


@dataclass
class FragmentSpec:
    """Fragment code generator.

    Example:

        Define a FragmentSpec in your kernel factory's specs like this:
            FragmentSpec("Acc", "MMA_M16_N8_F32_C", "qn", "k2")

        Use the generated class in your CUDA kernel like this:
            # Get element coordinates for this thread.
            int lane = threadIdx.x % 32;
            Acc:Index acc_idx(lane);
            auto lane_k2 = acc_idx.k2();
            auto lane_qn_0 = acc_idx.qn(0);
            auto lane_qn_8 = acc_idx.qn(0);

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
    fragment_type: str
    row: str
    col: str

    def generate(self) -> str:
        """Generate the fragment class code."""
        index_class = self.generate_index()

        load_index_class = self.generate_load_index()

        return f"""
class {self.class_name} : public spio::{self.fragment_type} {{
    public:
        {index_class}

        {load_index_class}
}};
"""

    def generate_index(self) -> str:
        """Generate the fragment index class code."""
        return FragmentIndexSpec(
            "Index", self.fragment_type, self.row, self.col
        ).generate()

    def generate_load_index(self) -> str:
        """Generate the fragment load index class code."""
        if not fragment_load_supported(self.fragment_type):
            return ""
        return FragmentLoadIndexSpec(
            "LoadIndex", self.fragment_type, self.row, self.col
        ).generate()


def _fragment_header() -> str:
    return """
#include "spio/fragment.cuh"
#include "spio/fragment_index.h"
#include "spio/fragment_load_index.h"
"""
