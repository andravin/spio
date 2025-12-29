"""Code generator for matrix fragment with named dimensions."""

from typing import Tuple
from dataclasses import dataclass

from .fragment_type import FragmentType
from .gen_specs import GenSpecs
from .derived_dimension import DerivedDimension


@dataclass
class Fragment(GenSpecs):
    """Fragment code generator.

    Example:

        Define a Fragment spec in your kernel factory's specs like this:
            Fragment(FragmentType.M16_N8_F32_C, "qn", "k2")

        Use the generated class in your CUDA kernel like this:
            # Get element coordinates for this thread.
            int lane = threadIdx.x % 32;
            Acc:CompoundIndex acc_idx(lane);
            auto qn_val = acc_idx.get<QN>();
            auto k2_val = acc_idx.get<K2>();

            # Define an accumulator and initialize it to zero.
            Acc acc;
            acc.zero();

    When used with the Generators container, class_name can be omitted and will
    be set from the attribute name.

    Attributes:
        fragment_type: Type of the fragment (see spio.include.spio / fragment.cuh)
        row: Name of the row dimension.
        col: Name of the column dimension.
        class_name: Name of the fragment class (optional with Generators).
    """

    fragment_type: FragmentType
    row: str
    col: str
    class_name: str = None

    def __post_init__(self):
        """Normalize the row and column dimension names to upper-case."""
        object.__setattr__(self, "row", self.row.upper())
        object.__setattr__(self, "col", self.col.upper())

    def _set_class_name(self, name: str) -> None:
        """Set the class name for this fragment.

        Called by the Generators container when the fragment is assigned to an attribute.
        """
        self.class_name = name

    def get_class_name(self) -> str:
        """Return the class name, or None if not set."""
        return self.class_name

    def generate(self) -> str:
        """Generate the fragment class code as a type alias."""
        row_dim = self.row
        col_dim = self.col
        fragment_class = f"spio::{self.fragment_type.value}<{row_dim}, {col_dim}>"
        return f"using {self.class_name} = {fragment_class};\n"

    @property
    def load_index(self) -> "FragmentLoadIndex":
        """Return the load index generator."""
        return FragmentLoadIndex(self)

    @property
    def compound_index(self) -> "FragmentCompoundIndex":
        """Return the compound index generator."""
        return FragmentCompoundIndex(self)

    @property
    def dim_names(self) -> Tuple[str, str]:
        """Return the names of the dimensions."""
        return (self.row, self.col)


class FragmentLoadIndex(GenSpecs, DerivedDimension):
    """Wrapper for a fragment load index type."""

    def __init__(self, fragment: Fragment):
        """Initialize the load index with the given fragment."""
        self.fragment = fragment

    def get_class_name(self) -> str:
        """Return the class name of the load index."""
        return self.fragment.get_class_name() + "::load_index_type"

    def used_generators(self) -> list[GenSpecs]:
        """Return a list of generator class-names used by this generator."""
        return []

    def generate(self) -> str:
        """Generate the load index type code as a type alias."""
        return ""  # Load index type is defined within the fragment class.


class FragmentCompoundIndex(GenSpecs, DerivedDimension):
    """Wrapper for a fragment compound index type."""

    def __init__(self, fragment: Fragment):
        """Initialize the compound index with the given fragment."""
        self.fragment = fragment

    def get_class_name(self) -> str:
        """Return the class name of the compound index."""
        return self.fragment.get_class_name() + "::compound_index_type"

    def used_generators(self) -> list[GenSpecs]:
        """Return a list of generator class-names used by this generator."""
        return []

    def generate(self) -> str:
        """Generate the compound index type code as a type alias."""
        return ""  # Compound index type is defined within the fragment class.`


def header() -> str:
    """Return the header file for the fragment classes."""
    return """
#include "spio/fragment.cuh"
"""
