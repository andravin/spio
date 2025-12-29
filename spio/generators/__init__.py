"""CUDA code generators."""

from .generators import generate
from .generators_class import Generators
from .gen_specs import GenSpecs
from .compound_index import CompoundIndex, CompoundIndexPartition
from .tensor import Tensor, CursorInitializer
from .fragment_type import FragmentType
from .data_type import dtype, get_dtype_veclen
from .fragment import Fragment
from .fragment_index import FragmentIndex, FragmentLoadIndex
from .macros import Macro
from .params import ParamsSpec
from .checkerboard import Checkerboard
from .async_strip_loader import AsyncStripLoader
from .dim import Dim, dim_name_to_dim_or_fold_class_name, BUILTIN_DIM_NAMES
from .fold import Fold
from .dims import Dims, Strides
from .matmul import Matmul
from .built_in import BuiltIn
from .coordinates import Coordinates
from .derived_dimension import DerivedDimension, SizedDerivedDimension

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
    "Coordinates",
    "CompoundIndexPartition",
]

__all__ = GENERATORS + [
    "generate",
    "Generators",
    "GenSpecs",
    "FragmentType",
    "dtype",
    "Checkerboard",
    "AsyncStripLoader",
    "dim_name_to_dim_or_fold_class_name",
    "Dims",
    "Strides",
    "BuiltIn",
]
