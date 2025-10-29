"""Setup script for the spio package."""

import os

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext


def _cuda_rt_include():
    # Import lazily so pip can read metadata without build deps installed
    from importlib_resources import files

    inc = files("nvidia.cuda_runtime").joinpath("include")
    inc_path = str(inc)
    if not os.path.isdir(inc_path):
        raise RuntimeError(f"CUDA runtime include dir not found in wheel: {inc_path}")

    # Ensure the driver API header we need is present
    if not os.path.isfile(os.path.join(inc_path, "cuda.h")):
        raise RuntimeError(
            "cuda.h not found in nvidia-cuda-runtime-cu12 include dir. "
            "Ensure a CUDA 12.x runtime wheel that ships driver headers is installed."
        )
    return inc_path


class build_ext(_build_ext):
    """Custom build_ext to define extensions lazily."""
    def build_extensions(self):
        from Cython.Build import cythonize

        inc_path = _cuda_rt_include()
        exts = [
            Extension(
                name="spio.cuda.driver",
                sources=["spio/cuda/driver.pyx"],
                # We use the CUDA Driver API at runtime: link to libcuda.so from the NVIDIA driver
                libraries=["cuda"],
                include_dirs=[inc_path],
                language="c",
            ),
        ]
        self.extensions = cythonize(exts, language_level=3)
        super().build_extensions()


setup(
    name="spio",
    version="0.3.0",
    packages=find_packages(),
    ext_modules=[],  # built in build_ext
    cmdclass={"build_ext": build_ext},
    include_package_data=True,
)
