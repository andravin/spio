"""Code generator for matrix fragment with named dimensions."""

from dataclasses import dataclass
from typing import Tuple

from .fragment_type import FragmentType
from .fragment_index import (
    FragmentLoadIndex,
    FragmentIndex,
    fragment_load_supported,
)


@dataclass
class Fragment:
    """Fragment code generator.

    Example:

        Define a FragmentSpec in your kernel factory's specs like this:
            FragmentSpec("Acc", "MMA_M16_N8_F32_C", "qn", "k2")

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
        """Generate the fragment class code."""
        index_class = self.generate_index()
        load_index_class = self.generate_load_index()

        if load_index_class:
            load_method = f"""
        __device__ static {self.class_name} from_smem(const void *p) {{ 
            {self.class_name} f;
            f.load(p);
            return f;
        }}

        __device__ static {self.class_name} from_smem_trans(const void *p) {{ 
            {self.class_name} f;
            f.load_trans(p);
            return f;
        }}"""
        else:
            load_method = ""

        return f"""
class {self.class_name} : public spio::{self.fragment_type.value} {{
    public:
        {index_class}
        {load_index_class}
        {load_method}
}};"""

    @property
    def dim_names(self) -> Tuple[str, str]:
        """Return the names of the dimensions."""
        return (self.row, self.col)

    def generate_index(self) -> str:
        """Generate the fragment index class code."""
        return FragmentIndex("Index", self.fragment_type, self.row, self.col).generate()

    def generate_load_index(self) -> str:
        """Generate the fragment load index class code."""
        if not fragment_load_supported(self.fragment_type):
            return ""
        return FragmentLoadIndex(
            "LoadIndex", self.fragment_type, self.row, self.col
        ).generate()


def header() -> str:
    return """
#include "spio/fragment.cuh"
#include "spio/fragment_index.h"
#include "spio/fragment_load_index.h"
"""
