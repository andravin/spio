"""Code generator for custom tensor classes in CUDA source code."""

from typing import Dict, Union, Generator, List
from dataclasses import dataclass

from .dim import dim_name_to_dim_or_fold_class_name
from .dims import Dims, Strides, compute_full_strides
from .fragment_type import FragmentType
from .data_type import dtype


@dataclass
class Tensor:
    """CUDA Code generator for custom tensor classes.

    This class is used to generate custom tensor classes that map named tensor
    dimensions to pointers.

    The user may optionally optionally set the stride of a dimension
    by specifying it in the strides parameter. Any unspecified stride is automatically
    calculated as the the size of the next dimension times the stride of the next dimension,
    with the last dimension having a stride of 1.

    Attributes:
        class_name (str): The name of the custom tensor class.
        data_type (Union[dtype, FragmentType]): The data type of the tensor elements.
        dims (Dims): A dictionary mapping dimension names to their sizes.
        strides (Strides): An optional dictionary mapping dimension names to their strides.
    """

    class_name: str
    data_type: Union[dtype, FragmentType]
    dims: Dims
    strides: Strides = None
    constant: bool = False

    def __post_init__(self):
        self.strides = compute_full_strides(self.dims, self.strides)

    def generate_with_context(self, user_data_types: List[str] = None) -> str:
        """Generate the C++ source code for the custom tensor class."""
        data_type_name = self._get_data_type_name(user_data_types=user_data_types)
        return _generate_tensor(
            self.class_name,
            data_type_name,
            self.dims,
            self.strides,
        )

    @property
    def size(self) -> int:
        """The number of elements required to store the tensor data."""
        name_0, size_0 = next(iter(self.dims.items()))
        stride_0 = self.strides[name_0]
        return size_0 * stride_0

    @property
    def num_bytes(self) -> int:
        """The number of bytes required to store the tensor data."""
        if isinstance(self.data_type, dtype):
            element_size = self.data_type.value.size
        else:
            raise ValueError(f"Size of data_type {self.data_type} not supported.")

        return self.size * element_size

    @property
    def dim_names(self) -> Generator[str, None, None]:
        """Return the names of the dimensions in the tensor."""
        for name, _ in self.dims.items():
            yield name

    def _get_data_type_name(self, user_data_types: List[str]) -> str:
        """Return the type-name for the tensor data type.

        The type-name is the literal name of the data type used in CUDA / C++ code.
        """
        return _get_data_type_name(
            self.data_type, constant=self.constant, user_data_types=user_data_types
        )


def _generate_tensor(
    class_name: str,
    data_type_name: str,
    dims: Dims,
    strides: Dict[str, int],
) -> str:
    """Generate a using statement for a Tensor template instantiation."""
    dim_infos = []

    for name, size_value in dims.items():
        size_str = str(size_value)
        stride = strides[name]
        dim_class = dim_name_to_dim_or_fold_class_name(name)
        dim_infos.append(f"spio::DimInfo<{dim_class}, {size_str}, {stride}>")

    # More concise formatting with line breaks only for longer declarations
    if len(dim_infos) <= 3:
        dim_info_str = ", ".join(dim_infos)
        return f"using {class_name} = spio::Tensor<{data_type_name}, {dim_info_str}>;\n"
    else:
        dim_info_str = ",\n    ".join(dim_infos)
        return f"using {class_name} = spio::Tensor<{data_type_name},\n    {dim_info_str}\n>;\n"


def header():
    """The C++ statement that includes the spio tensor header file.

    This file implements the C++ base template classes from which the
    custom tensor classes inherit. You must include this header before
    using the code returned by the generate_tensor() function.
    """
    return '#include "spio/tensor.h"'


def _get_data_type_name(
    data_type: Union[dtype, FragmentType],
    constant: bool = False,
    user_data_types: List[str] = None,
) -> str:
    if isinstance(data_type, FragmentType):
        data_type = f"spio::{data_type.value}"
    elif isinstance(data_type, dtype):
        data_type = data_type.value.name
    elif isinstance(data_type, str):
        if user_data_types is None:
            raise ValueError("user_data_types must be provided for user-defined data-types.")
        if not data_type in user_data_types:
            raise ValueError(f"Unknown user-defined data-type: {data_type}")
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
    if constant:
        data_type = f"const {data_type}"
    return data_type
