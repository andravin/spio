"""Code generator for fragment index that maps lanes to named dimensions."""

from dataclasses import dataclass


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
    "MMA_M16_K8_F16_A": _Desc(
        "MMA_A_88_F16_Index", "MMA_A_M16_K8_F16_LoadIndex", "i", "k", 2
    ),
    "MMA_M16_K16_F16_A": _Desc(
        "MMA_A_88_F16_Index", "MMA_A_M16_K16_F16_LoadIndex", "i", "k", 4
    ),
    "MMA_N8_K8_F16_B": _Desc(
        "MMA_B_88_F16_Index", "MMA_B_N8_K8_F16_LoadIndex", "k", "j", 1
    ),
    "MMA_N8_K16_F16_B": _Desc(
        "MMA_B_88_F16_Index", "MMA_B_N8_K16_F16_LoadIndex", "k", "j", 2, col_major=True
    ),
    "MMA_N16_K16_F16_B": _Desc(
        "MMA_B_88_F16_Index", "MMA_B_N16_K16_F16_LoadIndex", "k", "j", 4, col_major=True
    ),
    "MMA_M16_N8_F32_C": _Desc("MMA_C_88_F32_Index", None, "i", "j", 2),
    "MMA_M16_N16_F32_C": _Desc("MMA_C_88_F32_Index", None, "i", "j", 4),
}


class FragmentIndexSpec:
    """Fragment index code generator for matrix fragment with named dimensions.

    This class generates a subclass of the given fragment index type that adds
    methods for accessing the row and column index values using custom names
    for the row and column dimensions.
    """

    def __init__(
        self, class_name: str, fragment_type: str, row_name: str, col_name: str
    ):
        """Initialize the fragment index code generator.

        Args:
            class_name (str): The name of the class to generate.
            fragment_type (str): The fragment type.
            row_name (str): The name to use for the row index.
            col_name (str): The name to use for the column index.
        """
        self.class_name = class_name
        self.fragment_type = fragment_type
        self.row_name = row_name
        self.col_name = col_name

    def _major_axis(self, col_major: bool) -> str:
        """Return the name of the major axis."""
        return self.col_name if col_major else self.row_name

    def _minor_axis(self, col_major: bool) -> str:
        """Return the name of the minor axis."""
        return self.row_name if col_major else self.col_name

    def generate(self) -> str:
        """Return the CUDA / C++ source code for the fragment index subclass."""
        desc = _get_fragment_descriptor(self.fragment_type)
        base_major = desc.major_axis
        base_minor = _decorate_dim_name(desc.minor_axis, INDEX_MINOR_AXIS_VECLEN)
        base_minor_fragment = _decorate_dim_name(
            desc.minor_axis, INDEX_MINOR_AXIS_FRAGMENT_SIZE
        )
        base_minor_mod = _decorate_dim_name(
            desc.minor_axis, INDEX_MINOR_AXIS_VECLEN, MINOR_AXIS_VECS_PER_FRAGMENT
        )
        major = self._major_axis(desc.col_major)
        minor_name = self._minor_axis(desc.col_major)
        minor = _decorate_dim_name(minor_name, INDEX_MINOR_AXIS_VECLEN)
        minor_fragment = _decorate_dim_name(minor_name, INDEX_MINOR_AXIS_FRAGMENT_SIZE)
        minor_mod = _decorate_dim_name(
            minor_name, INDEX_MINOR_AXIS_VECLEN, MINOR_AXIS_VECS_PER_FRAGMENT
        )
        return f"""
class {self.class_name} : public spio::{desc.fragment_index} {{
    public:
        using Base = spio::{desc.fragment_index};
        using Base::Base;
        DEVICE constexpr int {major}(int idx = 0) const {{ return Base::{base_major}(idx); }}
        DEVICE constexpr int {minor}(int idx = 0) const {{ return Base::{base_minor}(idx); }}
        DEVICE constexpr int {minor_fragment}(int idx = 0) const {{ return Base::{base_minor_fragment}(idx); }}
        DEVICE constexpr int {minor_mod}() const {{ return Base::{base_minor_mod}(); }}
        DEVICE static constexpr int size() {{ return {desc.num_fragments}; }}
}};
"""


class FragmentLoadIndexSpec(FragmentIndexSpec):
    """Fragment load index code generator for matrix fragment with named dimensions.

    This class generates a subclass of the given fragment's load index that adds
    methods for accessing the row and column index values using custom names for
    the row and column dimensions.
    """

    def generate(self) -> str:
        """Return the CUDA / C++ source code for the fragment index subclass."""
        desc = _get_fragment_descriptor(self.fragment_type)
        base_major = desc.major_axis
        base_minor = _decorate_dim_name(desc.minor_axis, LOAD_INDEX_MINOR_AXIS_VECLEN)
        major = self._major_axis(desc.col_major)
        minor_name = self._minor_axis(desc.col_major)
        minor = _decorate_dim_name(minor_name, LOAD_INDEX_MINOR_AXIS_VECLEN)
        return f"""
class {self.class_name} : public spio::{desc.fragment_load_index} {{
    public:
        using Base = spio::{desc.fragment_load_index};
        using Base::Base;
        DEVICE constexpr int {major}() const {{ return Base::{base_major}(); }}
        DEVICE constexpr int {minor}() const {{ return Base::{base_minor}(); }}
}};
"""


def fragment_load_supported(fragment_type: str) -> bool:
    """Return True if the fragment type supports loading, False otherwise."""
    desc = _get_fragment_descriptor(fragment_type)
    return desc.fragment_load_index is not None


def _get_fragment_descriptor(fragment_type: str) -> _Desc:
    """Return the specification of a given fragment type."""
    desc = FRAGMENT_DESCRIPTORS.get(fragment_type, None)
    if desc is None:
        raise ValueError(f"Unsupported fragment type: {fragment_type}")
    return desc


def _decorate_dim_name(name: str, veclen: int = None, mod: int = None) -> str:
    """Return the decorated dimension name."""
    code = name
    if veclen is not None:
        code += f"{veclen}"
    if mod is not None:
        code += f"m{mod}"
    return code


def _fragment_index_header() -> str:
    """Return the C++ source code that tests a custom index class."""
    return """
#include "spio/fragment_index.h"
"""


def _fragment_load_index_header() -> str:
    """Return the C++ source code that tests a custom index class."""
    return """
#include "spio/fragment_load_index.h"
"""
