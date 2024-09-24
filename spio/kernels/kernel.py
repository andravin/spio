from tempfile import NamedTemporaryFile

import torch

from ..generators import generate
from ..compiler import compile_kernel, load_kernel
from .launch_params import LaunchParams
from .kernel_util import get_first_device_in_args
from ..cuda import primary_context_guard
from .stats import Stats


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
            None,
            self.cubin_file.name,
            {"parameters.h": self.parameters_header},
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
        self.function = None
        self.cubin_file = NamedTemporaryFile(
            suffix=".cubin",
            prefix=f"spio_{self.kernel_name}_",
        )
        self.parameters_header = generate(specs)

    def compile(self):
        return compile_kernel(*self.compiler_args)

    def load(self, device_ordinal=0):
        self.module, self.function = load_kernel(
            kernel_name=self.kernel_name,
            cubin_file_name=self.cubin_file.name,
            device_ordinal=device_ordinal,
        )
        self.cubin_file = None

    def unload(self):
        if self.module is not None:
            self.module.unload()
            self.module = None
        self.function = None

    def launch(self, *args):
        try:
            device = get_first_device_in_args(args)
            _check_args(args, device)
            kernel_args = _kernel_args(args)
            primary_context_guard.set_device(device.index)
            self.function.launch(
                self.launch_params.grid, self.launch_params.block, kernel_args
            )
        except Exception as e:
            raise ValueError(f"Error in kernel {self}") from e

    @property
    def stats(self):
        return self.Stats(params=self.params, unit=2, output_names=self.output_names)

    def __call__(self, *args):
        self.launch(*args)

    def __repr__(self) -> str:
        return f"{self.kernel_name} {self.params} {self.config}"

    @property
    def is_backprop(self):
        return any(
            [output_name.startswith("grad_") for output_name in self.output_names]
        )


def _check_args(args, device):
    """Ensure that all tensor arguments are on the same device and are channels_last contiguous."""
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.numel() > 0:
            assert (len(arg.shape) != 4) or arg.is_contiguous(
                memory_format=torch.channels_last
            ), f"Not all tensors arguments are channels_last contiguous: {args}"
            assert (
                device == arg.device
            ), f"Not all tensor arguments are on the same device: {args}"


def _kernel_args(args):
    return [t.detach() if isinstance(t, torch.Tensor) else t for t in args]
