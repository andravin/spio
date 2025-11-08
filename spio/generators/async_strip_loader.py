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
    """

    class_name: str
    smem_tensor: Tensor
    gmem_tensor: Tensor
    minor_axis: str
    major_axis_size: int
    minor_axis_size: int
    num_warps: int

    def generate(self) -> str:
        """Generate the C++ source code for the custom strip loader class."""
        smem_stride = self.smem_tensor.strides[self.minor_axis]
        gmem_stride = self.gmem_tensor.strides[self.minor_axis]
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
