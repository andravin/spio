"""Code generator for a custom strip loader class in CUDA / C++."""

from dataclasses import dataclass

from .gen_specs import GenSpecs
from .tensor import Tensor


@dataclass
class AsyncStripLoader(GenSpecs):
    """CUDA Code generator for custom strip loader classes.

    This class is used to generate custom strip loader classes that load
    data asynchronously from a global memory tensor to a shared memory
    tensor.

    When used with the Generators container, class_name can be omitted and will
    be set from the attribute name.

    Attributes:
        smem_tensor: The shared memory tensor.
        gmem_tensor: The global memory tensor.
        minor_axis: The minor axis name.
        major_axis_size: The major axis size.
        minor_axis_size: The minor axis size.
        num_warps: Number of warps.
        class_name: The name of the generated class (optional with Generators).
    """

    smem_tensor: Tensor
    gmem_tensor: Tensor
    minor_axis: str
    major_axis_size: int
    minor_axis_size: int
    num_warps: int
    class_name: str = None

    def __post_init__(self):
        """Normalize the minor axis name to upper-case."""
        self.minor_axis = self.minor_axis.upper()

    def _set_class_name(self, name: str) -> None:
        """Set the class name for this loader.

        Called by the Generators container when assigned to an attribute.
        """
        self.class_name = name

    def get_class_name(self) -> str:
        """Return the class name, or None if not set."""
        return self.class_name

    def generate(self) -> str:
        """Generate the C++ source code for the custom strip loader class."""
        smem_stride = _resolve_tensor(self.smem_tensor).strides[self.minor_axis]
        gmem_stride = _resolve_tensor(self.gmem_tensor).strides[self.minor_axis]
        params = self._gen_strip_loader_params()
        base_params = _make_args_list(smem_stride, gmem_stride, f"{params}::num_loads")
        base = f"spio::AsyncStripLoader<{base_params}>"
        return f"""
class {self.class_name} : public {base}
{{
    static constexpr int active_warps = {params}::active_warps;
    using Base = {base};
    using Base::Base;
}};
"""

    def _gen_strip_loader_params(self) -> str:
        pars = _make_args_list(
            self.major_axis_size, self.minor_axis_size, self.num_warps
        )
        return f"spio::StripLoaderParams<{pars}>"


def _make_args_list(*args):
    sep = ", "
    return sep.join(str(arg) for arg in args)


def _resolve_tensor(tensor) -> Tensor:
    """Helper to get the tensor from a Tensor or TensorRef."""
    return tensor.tensor if hasattr(tensor, "tensor") else tensor
