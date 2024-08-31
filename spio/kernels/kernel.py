from tempfile import NamedTemporaryFile

import cupy as cp

from ..generators import generate
from ..compiler import compile_kernel, load_kernel
from .launch_params import LaunchParams
from .code_directory import GenDirectory


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

    def __init__(self, kernel_name:str, launch_params:LaunchParams, kernel_source_file=None, specs=[]):
        self._kernel_source_file = kernel_source_file
        self.kernel_name = kernel_name
        self.launch_params = launch_params
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

    def launch(self, *tensors):
        cp_tensors = [cp.asarray(t) for t in tensors]
        self.kernel(self.launch_params.grid, self.launch_params.block, cp_tensors)

    def __call__(self, *tensors):
        self.launch(*tensors)
