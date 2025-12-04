"""CUDA code generators."""

from .generators import generate
from .gen_specs import GenSpecs
from .compound_index import CompoundIndex
from .tensor import Tensor
from .fragment_type import FragmentType
from .data_type import dtype
from .fragment import Fragment
from .fragment_index import FragmentIndex, FragmentLoadIndex
from .macros import Macro
from .params import ParamsSpec
from .checkerboard import Checkerboard
from .async_strip_loader import AsyncStripLoader
from .dim import Dim, dim_name_to_dim_or_fold_class_name
from .fold import Fold
from .dims import Dims, Strides
from .matmul import Matmul

GENERATORS = [
    "Tensor",
    "CompoundIndex",
    "Fragment",
    "Macro",
    "Dim",
    "Fold",
    "FragmentIndex",
    "FragmentLoadIndex",
    "ParamsSpec",
    "Matmul",
]

__all__ = GENERATORS + [
    "generate",
    "GenSpecs",
    "FragmentType",
    "dtype",
    "Checkerboard",
    "AsyncStripLoader",
    "dim_name_to_dim_or_fold_class_name",
    "Dims",
    "Strides",
]
