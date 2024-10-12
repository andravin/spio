import torch

from .. import primary_context_guard
from ..generators import generate
from ..compiler import compile_kernel, load_kernel
from .launch_params import LaunchParams
from .kernel_util import get_first_device_in_args
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
        self.parameters_header = generate(specs)
        self.cubin = None

    def compile(self):
        self.cubin = compile_kernel(*self.compiler_args)

    def load(self, device_ordinal=0, clear_cubin=True):
        """Load the compile kernel binary into a device.

        Args:
            device_ordinal (int, optional): The device ordinal to load the kernel onto. Defaults to 0.
            clear_cubin (bool, optional): Whether to clear the kernel binary after loading. Defaults to True.
        """
        self.module, self.function = load_kernel(
            kernel_name=self.kernel_name,
            cubin=self.cubin,
            device_ordinal=device_ordinal,
        )
        if clear_cubin:
            self.cubin = None
        self.parameters_header = None

    def unload(self):
        """Unload the kernel from the device.

        If you previously called the load method with clear_cubin=False,
        you can load the kernel again without recompiling it.
        """
        if self.module is not None:
            self.module.unload()
            self.module = None
        self.function = None

    def launch(self, *args):
        """Launch the kernel with the given arguments."""
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

    @classmethod
    def get_kernel_name(cls, params=None, **kwargs) -> str:
        """Return the full kernel name for the given parameters.

        The full name includes the base name and the encoded parameters.
        """
        base_name = cls.get_kernel_base_name(**kwargs)
        details = params.encode()
        return f"{base_name}__{details}"

    def __call__(self, *args):
        self.launch(*args)

    def __repr__(self) -> str:
        return f"{self.kernel_name} {self.params} {self.config}"


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
