"""Code generator for matrix fragment with named dimensions."""

from dataclasses import dataclass


@dataclass
class FragmentSpec:
    """Fragment code generator.

    Example:

        Define a FragmentSpec in your kernel factory's specs like this:
            FragmentSpec("Acc", "MMA_M16_N8_F32_C", "qn", "k")

        Use the generated class in your CUDA kernel like this:
            # Get element coordinates for this thread.
            auto lane_k2 = Acc::k2(idx.lane());
            auto lane_qn_0 = Acc::qn(idx.lane(), 0);
            auto lane_qn_8 = Acc::qn(idx.lane(), 1);

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
        return f"""
class {self.class_name} : public spio::{self.fragment_type} {{
    public:
        DEVICE static constexpr int {self.row}(unsigned lane_id, int idx) {{ return row(lane_id, idx); }}
        DEVICE static constexpr int {self.col}2(unsigned lane_id) {{ return col2(lane_id); }}
        DEVICE static constexpr int {self.col}(unsigned lane_id) {{ return col(lane_id); }}
}};
"""


def _fragment_header() -> str:
    return """
#include "spio/mma.cuh"
"""


def _start_namespace(namespace: str) -> str:
    return f"""

namespace {namespace} {{
"""


def _end_namespace() -> str:
    return """
}
"""
