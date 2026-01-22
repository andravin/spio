"""Define the fragment type enumeration."""

from enum import Enum
from dataclasses import dataclass

from .data_type import dtype


class Operand(Enum):
    """Operand type enumeration."""

    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True)
class _FragmentSpecs:
    """Specifications for a matrix fragment."""

    operand: Operand
    num_rows: int
    num_cols: int
    dtype: dtype


@dataclass(frozen=True)
class _FragmentInfo(_FragmentSpecs):
    """Fragment class_name and specifications."""

    class_name: str

    @property
    def specs(self) -> _FragmentSpecs:
        """Return the fragment specifications."""
        return _FragmentSpecs(self.operand, self.num_rows, self.num_cols, self.dtype)


class FragmentType(Enum):
    """Fragment type enumeration.

    Use one of these constants for any spio function that requests a fragment type.
    The fragment types correspond with those documented for the mma instruction
    in PTX ISA [1].

    Current support includes fragments types for float16 multiplication with float32 accumulation.

    # pylint: disable=line-too-long
    [1] https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma
    """

    M16_N8_F32_C = _FragmentInfo(Operand.C, 16, 8, dtype.float, "MMA_M16_N8_F32_C")
    M16_N16_F32_C = _FragmentInfo(Operand.C, 16, 16, dtype.float, "MMA_M16_N16_F32_C")
    M16_K8_F16_A = _FragmentInfo(Operand.A, 16, 8, dtype.half, "MMA_M16_K8_F16_A")
    M16_K16_F16_A = _FragmentInfo(Operand.A, 16, 16, dtype.half, "MMA_M16_K16_F16_A")
    N8_K8_F16_B = _FragmentInfo(Operand.B, 8, 8, dtype.half, "MMA_N8_K8_F16_B")
    N8_K16_F16_B = _FragmentInfo(Operand.B, 8, 16, dtype.half, "MMA_N8_K16_F16_B")
    N16_K16_F16_B = _FragmentInfo(Operand.B, 16, 16, dtype.half, "MMA_N16_K16_F16_B")


def _build_fragment_spec_type_lookup():
    fragment_type_lookup = {}
    for fragment_type in FragmentType:
        info = fragment_type.value
        fragment_type_lookup[info.specs] = fragment_type
    return fragment_type_lookup


_FRAGMENT_TYPE_LOOKUP = _build_fragment_spec_type_lookup()


def lookup_fragment_type(
    operand: Operand, data_type: dtype, num_rows: int, num_cols: int
) -> FragmentType:
    """Look up the fragment type based on operand, data type, and dimensions.

    Args:
        operand: The operand type (OperandType.A, .B, or .C).
        data_type: The data type of the fragment.
        num_rows: The number of rows in the fragment.
        num_cols: The number of columns in the fragment.

    Returns:
        The corresponding FragmentType.

    Raises:
        TypeError: If data_type is not a dtype enum value.
        ValueError: If no matching fragment type is found.
    """
    if not isinstance(data_type, dtype):
        raise TypeError(f"data_type must be a dtype enum value, got {type(data_type).__name__}")
    specs = _FragmentSpecs(operand, num_rows, num_cols, data_type)
    if specs not in _FRAGMENT_TYPE_LOOKUP:
        raise ValueError(
            f"No fragment type found for operand={operand.value}, data_type={data_type.name}, "
            f"num_rows={num_rows}, num_cols={num_cols}."
        )
    return _FRAGMENT_TYPE_LOOKUP[specs]
