from tempfile import NamedTemporaryFile

import cupy as cp

from ..compile_kernel import compile_kernel, load_kernel
from spio import generate, GenDirectory


class Kernel:
    @property
    def kernel_source_file(self):
        return f"{self.kernel_name}.cu"

    @property
    def compiler_args(self):
        return (
            self.kernel_name,
            self.kernel_source_file,
            self.debug,
            self.lineinfo,
            [self.gencode_dir.name],
            self.cubin_file.name,
        )

    def __init__(
        self,
        debug=False,
        lineinfo=False,
    ):
        self.module = None
        self.kernel = None
        self.debug = debug
        self.lineinfo = lineinfo

    def generate(self, specs):
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
