"""Setup script for the spio package."""

import importlib.resources

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


def _get_cuda_rt_include_path() -> str:
    """Get the CUDA runtime include path from the nvidia.cuda_runtime package."""
    try:
        with importlib.resources.files("nvidia.cuda_runtime.include") as path:
            return str(path)
    except FileNotFoundError as e:
        raise RuntimeError("Could not find CUDA runtime include directory.") from e


extensions = [
    Extension(
        name="spio.cuda.driver",
        sources=["spio/cuda/driver.pyx"],
        libraries=["cuda"],
        include_dirs=[_get_cuda_rt_include_path()],
    ),
]


setup(
    name="spio",
    version="0.1.0rc1",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    install_requires=[
        "torch>=2.4.0",
        "nvidia-cuda-nvrtc-cu12",
        "pytest",
        "xgboost",
        "appdirs",
        "requests",
        "filelock",
        "packaging",
    ],
    include_package_data=True,
)
