"""Define the fragment type enumeration."""

from enum import Enum
from dataclasses import dataclass

from .data_type import dtype


@dataclass
class _FragmentSpecs:
    """Specifications for a matrix fragment."""

    class_name: str
    num_rows: int
    num_cols: int
    dtype: dtype


class FragmentType(Enum):
    """Fragment type enumeration.

    Use one of these constants for any spio function that requests a fragment type.
    The fragment types correspond with those documented for the mma instruction
    in PTX ISA [1].

    Current support includes fragments types for float16 multiplication with float32 accumulation.

    [1] https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-multiply-accumulate-operation-using-mma-instruction
    """

    M16_N8_F32_C = _FragmentSpecs("MMA_M16_N8_F32_C", 16, 8, dtype.float)
    M16_N16_F32_C = _FragmentSpecs("MMA_M16_N16_F32_C", 16, 16, dtype.float)
    M16_K8_F16_A = _FragmentSpecs("MMA_M16_K8_F16_A", 16, 8, dtype.half)
    M16_K16_F16_A = _FragmentSpecs("MMA_M16_K16_F16_A", 16, 16, dtype.half)
    N8_K8_F16_B = _FragmentSpecs("MMA_N8_K8_F16_B", 8, 8, dtype.half)
    N8_K16_F16_B = _FragmentSpecs("MMA_N8_K16_F16_B", 8, 16, dtype.half)
    N16_K16_F16_B = _FragmentSpecs("MMA_N16_K16_F16_B", 16, 16, dtype.half)
