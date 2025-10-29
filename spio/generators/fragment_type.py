"""Define the fragment type enumeration."""
from enum import Enum


class FragmentType(Enum):
    """Fragment type enumeration.

    Use one of these constants for any spio function that requests a fragment type.
    The fragment types correspond with those documented for the mma instruction
    in PTX ISA [1].

    Current support includes fragments types for float16 multiplication with float32 accumulation.

    [1] https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-multiply-accumulate-operation-using-mma-instruction
    """

    M16_N8_F32_C = "MMA_M16_N8_F32_C"
    M16_N16_F32_C = "MMA_M16_N16_F32_C"
    M16_K8_F16_A = "MMA_M16_K8_F16_A"
    M16_K16_F16_A = "MMA_M16_K16_F16_A"
    N8_K8_F16_B = "MMA_N8_K8_F16_B"
    N8_K16_F16_B = "MMA_N8_K16_F16_B"
    N16_K16_F16_B = "MMA_N16_K16_F16_B"
