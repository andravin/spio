"""Code generator for fragment index that maps lanes to named dimensions."""

from dataclasses import dataclass
from typing import Tuple

from .dim import dim_name_to_dim_or_fold_class_name
from .fragment_type import FragmentType


@dataclass
class _Desc:
    """Descriptor for a fragment type.

    Args:
        fragment_index (str): The index type name.
        fragment_load_index (str): The load index type name or None.
        row (str): The row name.
        col (str): The column name.
        col_major (bool): True if the column is the major axis, False otherwise.
    """

    fragment_index: str
    fragment_load_index: str
    row: str
    col: str
    num_fragments: int
    col_major: bool = False

    @property
    def major_axis(self):
        """Return the name of the major axis."""
        return self.col if self.col_major else self.row

    @property
    def minor_axis(self):
        """Return the name of the minor axis."""
        return self.row if self.col_major else self.col


INDEX_MINOR_AXIS_VECLEN = 2
INDEX_MINOR_AXIS_FRAGMENT_SIZE = 8
LOAD_INDEX_MINOR_AXIS_VECLEN = 8
MINOR_AXIS_VECS_PER_FRAGMENT = INDEX_MINOR_AXIS_FRAGMENT_SIZE // INDEX_MINOR_AXIS_VECLEN


FRAGMENT_DESCRIPTORS = {
    FragmentType.M16_K8_F16_A: _Desc(
        "MMA_A_88_F16_Index", "MMA_A_M16_K8_F16_LoadIndex", "i", "k", 2
    ),
    FragmentType.M16_K16_F16_A: _Desc(
        "MMA_A_88_F16_Index", "MMA_A_M16_K16_F16_LoadIndex", "i", "k", 4
    ),
    FragmentType.N8_K8_F16_B: _Desc(
        "MMA_B_88_F16_Index", "MMA_B_N8_K8_F16_LoadIndex", "k", "j", 1, col_major=True
    ),
    FragmentType.N8_K16_F16_B: _Desc(
        "MMA_B_88_F16_Index", "MMA_B_N8_K16_F16_LoadIndex", "k", "j", 2, col_major=True
    ),
    FragmentType.N16_K16_F16_B: _Desc(
        "MMA_B_88_F16_Index", "MMA_B_N16_K16_F16_LoadIndex", "k", "j", 4, col_major=True
    ),
    FragmentType.M16_N8_F32_C: _Desc("MMA_C_88_F32_Index", None, "i", "j", 2),
    FragmentType.M16_N16_F32_C: _Desc("MMA_C_88_F32_Index", None, "i", "j", 4),
}


class FragmentIndex:
    """Fragment index code generator for matrix fragment with named dimensions.

    This class generates a type alias for the fragment index template class
    with the appropriate dimension types.
    """

    def __init__(
        self, class_name: str, fragment_type: FragmentType, row_name: str, col_name: str
    ):
        """Initialize the fragment index code generator.

        Args:
            class_name (str): The name of the class to generate.
            fragment_type (FragmentType): The fragment type.
            row_name (str): The name to use for the row index.
            col_name (str): The name to use for the column index.
        """
        self.class_name = class_name
        self.fragment_type = fragment_type
        self.row_name = row_name
        self.col_name = col_name

    def generate(self) -> str:
        """Return the CUDA / C++ source code for the fragment index type alias."""
        desc = _get_fragment_descriptor(self.fragment_type)

        # Get the class names for the dimension types
        row_dim_class = dim_name_to_dim_or_fold_class_name(self.row_name)
        col_dim_class = dim_name_to_dim_or_fold_class_name(self.col_name)

        # Generate the type alias
        return f"using {self.class_name} = spio::{desc.fragment_index}<{row_dim_class}, {col_dim_class}>;"

    @property
    def dim_names(self) -> Tuple[str, str]:
        """Return the names of the dimensions."""
        return (self.row_name, self.col_name, "lane")


class FragmentLoadIndex(FragmentIndex):
    """Fragment load index code generator for matrix fragment with named dimensions.

    This class generates a type alias for the fragment load index template class
    with the appropriate dimension types.
    """

    def generate(self) -> str:
        """Return the CUDA / C++ source code for the fragment load index type alias."""
        desc = _get_fragment_descriptor(self.fragment_type)

        if desc.fragment_load_index is None:
            return "// No load index available for this fragment type"

        # Get the class names for the dimension types
        row_dim_class = dim_name_to_dim_or_fold_class_name(self.row_name)
        col_dim_class = dim_name_to_dim_or_fold_class_name(self.col_name)

        # Generate the type alias
        return f"using {self.class_name} = spio::{desc.fragment_load_index}<{row_dim_class}, {col_dim_class}>;"


def fragment_load_supported(fragment_type: FragmentType) -> bool:
    """Return True if the fragment type supports loading, False otherwise."""
    desc = _get_fragment_descriptor(fragment_type)
    return desc.fragment_load_index is not None


def _get_fragment_descriptor(fragment_type: FragmentType) -> _Desc:
    """Return the specification of a given fragment type."""
    desc = FRAGMENT_DESCRIPTORS.get(fragment_type, None)
    if desc is None:
        raise ValueError(f"Unsupported fragment type: {fragment_type}")
    return desc


def header() -> str:
    """Return the C++ source code that tests a custom index class."""
    return """
#include "spio/fragment_index.h"
#include "spio/fragment_load_index.h"
"""
