"""Setup script for the spio package."""

import os
import subprocess
from pathlib import Path

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


def _find_libcuda_path() -> str:
    """Locate the CUDA driver shared library and return an absolute path."""
    # 1) Explicit file path override
    for env in ("SPIO_LIBCUDA", "CUDA_DRIVER_LIB"):
        p = os.environ.get(env)
        if p and Path(p).is_file():
            return str(Path(p).resolve())

    # 2) Directory override: look for libcuda.so or libcuda.so.1
    dir_env = os.environ.get("SPIO_LIBCUDA_DIR")
    if dir_env:
        for name in ("libcuda.so", "libcuda.so.1"):
            candidate = Path(dir_env) / name
            if candidate.is_file():
                return str(candidate.resolve())

    # 3) Parse ldconfig (Linux)
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, check=True).stdout
        # Prefer the generic soname if present, else .so.1
        preferred, fallback = None, None
        for line in out.splitlines():
            if "libcuda.so" in line:
                # format: "\tlibcuda.so.X (libc6,x86-64) => /path/libcuda.so.X"
                parts = line.split("=>")
                if len(parts) == 2:
                    path = parts[1].strip()
                    if path.endswith("libcuda.so"):
                        preferred = path
                    elif path.endswith("libcuda.so.1") and not fallback:
                        fallback = path
        if preferred:
            return preferred
        if fallback:
            return fallback
    except Exception:
        pass

    # 4) Common Linux locations
    for base in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/lib/x86_64-linux-gnu", "/lib64", "/lib"):
        for name in ("libcuda.so", "libcuda.so.1"):
            p = Path(base) / name
            if p.is_file():
                return str(p.resolve())

    raise RuntimeError(
        "libcuda not found. Install the NVIDIA driver, or set SPIO_LIBCUDA to the full path "
        "of libcuda.so(.1) or SPIO_LIBCUDA_DIR to its directory."
    )


class build_ext(_build_ext):
    """Custom build_ext to define extensions lazily."""
    def run(self):
        from Cython.Build import cythonize

        inc_path = _cuda_rt_include()
        libcuda_path = _find_libcuda_path()
        lib_dir = str(Path(libcuda_path).parent)

        # If we have libcuda.so, use -lcuda; if only libcuda.so.1, link by absolute path
        libraries = []
        library_dirs = []
        runtime_library_dirs = [lib_dir]
        extra_link_args = []
        if libcuda_path.endswith("libcuda.so"):
            libraries = ["cuda"]
            library_dirs = [lib_dir]
        else:
            # Link directly against soname file
            extra_link_args = [libcuda_path]

        exts = [
            Extension(
                name="spio.cuda.driver",
                sources=["spio/cuda/driver.pyx"],
                include_dirs=[inc_path],
                libraries=libraries,
                library_dirs=library_dirs,
                runtime_library_dirs=runtime_library_dirs,
                extra_link_args=extra_link_args,
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
    version="0.4.0",
    packages=find_packages(),
    # Keep a placeholder so build_ext is scheduled; link details are set in build_ext.run
    ext_modules=[Extension("spio.cuda.driver", sources=["spio/cuda/driver.pyx"])],
    cmdclass={"build_ext": build_ext},
    include_package_data=True,
)
