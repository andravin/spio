from tempfile import NamedTemporaryFile

import cupy as cp
import torch

from ..generators import generate
from ..compiler import compile_kernel, load_kernel
from .launch_params import LaunchParams
from .code_directory import GenDirectory
from .kernel_util import get_first_device_in_args


class Kernel:
    @property
    def kernel_source_file(self):
        if self._kernel_source_file is None:
            return f"{self.kernel_name}.cu"
        return self._kernel_source_file

    @property
    def compiler_args(self):
        return (
            self.kernel_name,
            self.kernel_source_file,
            [self.gencode_dir.name],
            self.cubin_file.name,
        )

    def __init__(
        self,
        kernel_name: str,
        launch_params: LaunchParams,
        kernel_source_file=None,
        specs=[],
        params=None,
        config=None,
    ):
        self._kernel_source_file = kernel_source_file
        self.kernel_name = kernel_name
        self.launch_params = launch_params
        self.params = params
        self.config = config
        self.module = None
        self.kernel = None
        self.cubin_file = NamedTemporaryFile(
            suffix=".cubin",
            prefix=f"spio_{self.kernel_name}_",
        )
        self.gencode_dir = GenDirectory()
        generate(
            specs,
            self.gencode_dir.path / "parameters.h",
        )

    def compile(self):
        res = compile_kernel(*self.compiler_args)
        self.gencode_dir.cleanup()
        self.gencode_dir = None
        return res

    def load(self):
        self.module, self.kernel = load_kernel(
            kernel_name=self.kernel_name, cubin_file_name=self.cubin_file.name
        )
        self.cubin_file = None

    def launch(self, *args):
        try:
            device = get_first_device_in_args(args)
            _check_args(args, device)
            cp_args = _detach_cupy_tensors(args)
            with torch.device(device):
                self.kernel(self.launch_params.grid, self.launch_params.block, cp_args)
        except Exception as e:
            raise ValueError(f"Error in kernel {self}") from e

    def __call__(self, *args):
        self.launch(*args)

    def __repr__(self) -> str:
        return f"{self.kernel_name} {self.params} {self.config}"


def _check_args(args, device):
    """Ensure that all tensor arguments are on the same device and are channels_last contiguous."""
    for arg in args:
        if isinstance(arg, torch.Tensor):
            assert arg.is_contiguous(
                memory_format=torch.channels_last
            ), f"Not all tensors arguments are channels_last contiguous: {args}"
        assert device == arg.device, f"Not all tensor arguments are on the same device: {args}"


def _detach_cupy_tensors(args):
    return [cp.asarray(t.detach()) if isinstance(t, torch.Tensor) else t for t in args]
