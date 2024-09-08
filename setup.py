from setuptools import Extension, setup
from Cython.Build import cythonize
import os
import sys


def _find_cuda_rt_include_in_sys_path():
    lib_path = None
    for path in sys.path:
        try_path = os.path.join(path, "nvidia", "cuda_runtime", "include")
        if os.path.exists(try_path):
            lib_path = try_path
            break
    return lib_path


cuda_rt_include_dir = _find_cuda_rt_include_in_sys_path()
if cuda_rt_include_dir is None:
    raise RuntimeError(
        "Could not find CUDA runtime include directory. Did you install PyTorch with CUDA support?"
    )

extensions = [
    Extension(
        name="spio.cuda.driver",
        sources=["spio/cuda/driver.pyx"],
        libraries=["cuda"],
        include_dirs=[cuda_rt_include_dir],
    ),
]


setup(
    ext_modules=cythonize(extensions),
)
