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
    def run(self):
        # Populate extensions before the parent run() checks them
        from Cython.Build import cythonize

        inc_path = _cuda_rt_include()
        exts = [
            Extension(
                name="spio.cuda.driver",
                sources=["spio/cuda/driver.pyx"],
                libraries=["cuda"],           # link to libcuda.so from the NVIDIA driver
                include_dirs=[inc_path],      # headers from nvidia-cuda-runtime wheel
                language="c",
            ),
        ]
        exts = cythonize(exts, language_level=3)

        # Editable builds with recent setuptools expect this attr on Extension
        for ext in exts:
            if not hasattr(ext, "_needs_stub"):
                ext._needs_stub = False

        self.extensions = exts
        super().run()


setup(
    name="spio",
    version="0.3.0",
    packages=find_packages(),
    # Make setuptools see an extension so build_ext is scheduled
    ext_modules=[Extension("spio.cuda.driver", sources=["spio/cuda/driver.pyx"])],
    cmdclass={"build_ext": build_ext},
    include_package_data=True,
)
