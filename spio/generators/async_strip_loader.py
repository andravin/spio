"""Code generator for a custom strip loader class in CUDA / C++."""

import re
from dataclasses import dataclass

from .gen_specs import GenSpecs
from .tensor import Tensor


@dataclass
class AsyncStripLoader(GenSpecs):
    """CUDA Code generator for async strip loader classes.

    This class generates custom strip loader classes that load data
    asynchronously from a global memory tensor to a shared memory tensor.

    The loader automatically chooses between 1D and 2D iteration based on
    whether num_warps covers the inner axis:
    - If num_warps >= inner_axis_size: uses 1D loader (iterate over outer axis only)
    - If num_warps < inner_axis_size: uses 2D loader (iterate over both axes)

    When used with the Generators container, class_name can be omitted and will
    be set from the attribute name.

    Attributes:
        smem_tensor: The shared memory tensor.
        gmem_tensor: The global memory tensor.
        inner_axis: The inner (major) axis base name (e.g., "i" or "j").
            The generator will find the matching dimension in each tensor.
        outer_axis: The outer (minor) axis name (e.g., "k16").
        inner_axis_size: Number of elements along the inner axis.
        outer_axis_size: Number of elements along the outer axis.
        num_warps: Number of warps cooperating in the load.
        class_name: The name of the generated class (optional with Generators).
    """

    smem_tensor: Tensor
    gmem_tensor: Tensor
    inner_axis: str
    outer_axis: str
    inner_axis_size: int
    outer_axis_size: int
    num_warps: int
    class_name: str = None
    num_buffers: int = 1

    def __post_init__(self):
        """Normalize axis names to upper-case."""
        self.inner_axis = self.inner_axis.upper()
        self.outer_axis = self.outer_axis.upper()

    def _set_class_name(self, name: str) -> None:
        """Set the class name for this loader."""
        self.class_name = name

    def get_class_name(self) -> str:
        """Return the class name, or None if not set."""
        return self.class_name

    def generate(self) -> str:
        """Generate the C++ source code for the custom strip loader class."""
        if self.num_warps >= self.inner_axis_size:
            return self._generate_1d()
        else:
            return self._generate_2d()

    def _generate_1d(self) -> str:
        """Generate 1D loader (iterate over outer axis only)."""
        smem = _resolve_tensor(self.smem_tensor)
        gmem = _resolve_tensor(self.gmem_tensor)

        smem_stride = smem.strides[self.outer_axis]
        gmem_stride = gmem.strides[self.outer_axis]

        smem_buffer_stride = smem_stride * self.outer_axis_size
        gmem_buffer_stride = gmem_stride * self.outer_axis_size

        data_type_name = smem.data_type.value.name

        params = self._gen_strip_loader_params()
        base_params = _make_args_list(
            self.smem_tensor.class_name,
            self.gmem_tensor.class_name,
            data_type_name,
            smem_stride,
            gmem_stride,
            f"{params}::num_loads",
            smem_buffer_stride,
            gmem_buffer_stride,
            self.num_buffers,
        )
        base = f"spio::AsyncStripLoader<{base_params}>"

        return f"""
class {self.class_name} : public {base}
{{
    static constexpr int active_warps = {params}::active_warps;
    using Base = {base};
    using Base::Base;
}};
"""

    def _generate_2d(self) -> str:
        """Generate 2D loader (iterate over both axes)."""
        smem = _resolve_tensor(self.smem_tensor)
        gmem = _resolve_tensor(self.gmem_tensor)

        # Find matching dimensions in each tensor for the inner axis
        smem_inner_axis = _find_axis(smem.strides, self.inner_axis)
        gmem_inner_axis = _find_axis(gmem.strides, self.inner_axis)

        smem_stride_inner = smem.strides[smem_inner_axis]
        gmem_stride_inner = gmem.strides[gmem_inner_axis]
        smem_stride_outer = smem.strides[self.outer_axis]
        gmem_stride_outer = gmem.strides[self.outer_axis]

        # num_inner = how many iterations along inner axis each warp does
        num_inner = (self.inner_axis_size + self.num_warps - 1) // self.num_warps
        if num_inner == 0:
            num_inner = 1
        num_outer = self.outer_axis_size

        # Inner stride: step the entire block by num_warps positions.
        smem_stride_inner_total = self.num_warps * smem_stride_inner

        # For gmem, account for fold factor between smem and gmem axes.
        smem_fold = _extract_fold_factor(smem_inner_axis)
        gmem_fold = _extract_fold_factor(gmem_inner_axis)
        fold_ratio = smem_fold // gmem_fold if gmem_fold > 0 else 1

        gmem_stride_inner_total = self.num_warps * fold_ratio * gmem_stride_inner

        smem_buffer_stride = smem_stride_outer * self.outer_axis_size
        gmem_buffer_stride = gmem_stride_outer * self.outer_axis_size // fold_ratio

        # Inner step dimension for cursor iteration (in gmem's units, e.g., I)
        inner_step_dim = gmem_inner_axis
        inner_step_size = self.num_warps * fold_ratio

        data_type = smem.data_type.value.name

        base_params = _make_args_list(
            self.smem_tensor.class_name,
            self.gmem_tensor.class_name,
            data_type,
            smem_stride_inner_total,
            gmem_stride_inner_total,
            num_inner,
            smem_stride_outer,
            gmem_stride_outer,
            num_outer,
            inner_step_dim,
            inner_step_size,
            smem_buffer_stride,
            gmem_buffer_stride,
            self.num_buffers,
        )
        base = f"spio::AsyncStripLoader2D<{base_params}>"

        return f"""
class {self.class_name} : public {base}
{{
    using Base = {base};
    using Base::Base;
}};
"""

    def _gen_strip_loader_params(self) -> str:
        pars = _make_args_list(
            self.inner_axis_size, self.outer_axis_size, self.num_warps
        )
        return f"spio::StripLoaderParams<{pars}>"


def _make_args_list(*args):
    sep = ", "
    return sep.join(str(arg) for arg in args)


def _resolve_tensor(tensor) -> Tensor:
    """Helper to get the tensor from a Tensor or TensorRef."""
    return tensor.tensor if hasattr(tensor, "tensor") else tensor


def _find_axis(strides: dict, base_name: str) -> str:
    """Find a dimension in strides dict that matches the base name.

    Looks for exact match first, then for dimensions starting with base_name.
    E.g., base_name="I" matches "I" or "I16".

    Returns the matching axis name (e.g., "I16" or "I").
    """
    # Exact match first
    if base_name in strides:
        return base_name

    # Look for dimensions starting with base_name (e.g., I16 for base I)
    candidates = [k for k in strides if k.startswith(base_name)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Prefer the one with largest fold factor (e.g., I16 over I8)
        return max(candidates, key=_extract_fold_factor)

    raise ValueError(
        f"No dimension matching '{base_name}' found in {list(strides.keys())}"
    )


def _extract_fold_factor(axis_name: str) -> int:
    """Extract the fold factor from an axis name like 'I16' -> 16, 'I' -> 1."""
    match = re.search(r"(\d+)$", axis_name)
    if match:
        return int(match.group(1))
    return 1
