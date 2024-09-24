from setuptools import setup, find_packages, Extension
import importlib.resources

from Cython.Build import cythonize


try:
    with importlib.resources.path("nvidia.cuda_runtime", "include") as path:
        cuda_rt_include_dir = str(path)
except FileNotFoundError as e:
    raise RuntimeError(
        "Could not find CUDA runtime include directory. Did you install PyTorch with CUDA support?"
    ) from e

extensions = [
    Extension(
        name="spio.cuda.driver",
        sources=["spio/cuda/driver.pyx"],
        libraries=["cuda"],
        include_dirs=[cuda_rt_include_dir],
    ),
]


setup(
    name="spio",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    install_requires=[
        "torch>=2.4.0",
        "nvidia-cuda-nvrtc-cu12",
        "pytest",
        "pandas",
        "xgboost",
        "scikit-learn",
        "appdirs",
    ],
    include_package_data=True,
)
