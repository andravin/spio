"""CUDA code generators."""

from .generators import generate
from .generators_class import Generators
from .gen_specs import GenSpecs
from .compound_index import CompoundIndex, CompoundIndexPartition
from .tensor import Tensor, CursorInitializer
from .fragment_type import FragmentType, Operand
from .data_type import dtype, get_dtype_veclen
from .fragment import Fragment, FragmentBase
from .fragment_index import FragmentIndex, FragmentLoadIndex
from .macros import Macro
from .params import ParamsSpec
from .checkerboard import Checkerboard
from .async_loader import AsyncLoader
from .dim import (
    Dim,
    StaticDim,
    dim_name_to_dim_or_fold_class_name,
    BUILTIN_DIM_NAMES,
    LANE,
    OFFSET,
)
from .dim_arg import DimArg, normalize_dim_arg
from .fold import Fold, StaticFold
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
    "FragmentBase",
    "dtype",
    "Checkerboard",
    "AsyncLoader",
    "dim_name_to_dim_or_fold_class_name",
    "Dims",
    "Strides",
    "BuiltIn",
    "LANE",
    "OFFSET",
]
