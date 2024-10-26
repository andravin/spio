"""Classes and functions for NVRTC (NVIDIA Runtime Compilation) using
ctypes."""

import ctypes
from typing import List, Tuple
from enum import Enum

from importlib_resources import files as importlib_resources_files


NVRTC_LIB = "libnvrtc.so.12"


def _find_libnvrtc() -> str:
    """Find the NVRTC shared lib in the nvidia.cuda_nvrtc package."""
    return str(importlib_resources_files("nvidia.cuda_nvrtc.lib").joinpath(NVRTC_LIB))


# Define the types.
class _nvrtc_Program(ctypes.Structure):
    """Opaque NVRTC program type."""

    _fields_ = []


nvrtc_Program = ctypes.POINTER(_nvrtc_Program)


def _define_nvrtc_types(nvrtc):
    """Define the function signatures."""
    nvrtc.nvrtcVersion.restype = ctypes.c_int
    nvrtc.nvrtcVersion.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]

    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
    nvrtc.nvrtcGetErrorString.argtypes = [ctypes.c_int]

    nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
    nvrtc.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(nvrtc_Program),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]

    nvrtc.nvrtcDestroyProgram.restype = ctypes.c_int
    nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(nvrtc_Program)]

    nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
    nvrtc.nvrtcCompileProgram.argtypes = [
        nvrtc_Program,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]

    nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int
    nvrtc.nvrtcGetPTXSize.argtypes = [nvrtc_Program, ctypes.POINTER(ctypes.c_size_t)]

    nvrtc.nvrtcGetPTX.restype = ctypes.c_int
    nvrtc.nvrtcGetPTX.argtypes = [nvrtc_Program, ctypes.c_char_p]

    nvrtc.nvrtcGetCUBINSize.restype = ctypes.c_int
    nvrtc.nvrtcGetCUBINSize.argtypes = [nvrtc_Program, ctypes.POINTER(ctypes.c_size_t)]

    nvrtc.nvrtcGetCUBIN.restype = ctypes.c_int
    nvrtc.nvrtcGetCUBIN.argtypes = [nvrtc_Program, ctypes.c_char_p]

    nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLogSize.argtypes = [
        nvrtc_Program,
        ctypes.POINTER(ctypes.c_size_t),
    ]

    nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLog.argtypes = [nvrtc_Program, ctypes.c_char_p]


try:
    lib_path = _find_libnvrtc()
except FileNotFoundError as e:
    raise ValueError(
        f"Could not find {NVRTC_LIB}. Did you install PyTorch with CUDA support?"
    ) from e
_nvrtc = ctypes.CDLL(lib_path)
_define_nvrtc_types(_nvrtc)


# Define the NVRTC error codes.
class NVRTCErrorCode(Enum):
    """NVRTC error codes."""

    NVRTC_SUCCESS = 0
    NVRTC_ERROR_OUT_OF_MEMORY = 1
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2
    NVRTC_ERROR_INVALID_INPUT = 3
    NVRTC_ERROR_INVALID_PROGRAM = 4
    NVRTC_ERROR_INVALID_OPTION = 5
    NVRTC_ERROR_COMPILATION = 6
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10
    NVRTC_ERROR_INTERNAL_ERROR = 11
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12


def version() -> Tuple[int, int]:
    """Return the NVRTC version as a (major, minor) tuple."""
    major = ctypes.c_int()
    minor = ctypes.c_int()
    result = _nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    if result != 0:
        raise ValueError("Failed to get NVRTC version")
    return major.value, minor.value


def error_string(code: int) -> str:
    """Return the error string for the given NVRTC error code."""
    return _nvrtc.nvrtcGetErrorString(code).decode("utf-8")


class NVRTCError(Exception):
    """Exception raised for NVRTC errors."""

    def __init__(self, err: int):
        self.err = err
        super().__init__(error_string(err))


def _check(err: int):
    """Check error and raise an exception if not NVRTC_SUCCESS."""
    if err != NVRTCErrorCode.NVRTC_SUCCESS.value:
        raise NVRTCError(err)


class Program:
    """NVRTC program class for compiling CUDA code."""

    def __init__(
        self,
        src: str,
        name: str,
        headers: List[str] = None,
        include_names: List[str] = None,
    ):
        """Create a new NVRTC program."""
        if headers is None:
            headers = []
        if include_names is None:
            include_names = []
        self.program = nvrtc_Program()
        num_headers = len(headers)
        headers_arr = (ctypes.c_char_p * num_headers)()
        headers_arr[:] = [header.encode("utf-8") for header in headers]
        include_names_arr = (ctypes.c_char_p * len(include_names))()
        include_names_arr[:] = [name.encode("utf-8") for name in include_names]
        _check(
            _nvrtc.nvrtcCreateProgram(
                ctypes.byref(self.program),
                src.encode("utf-8"),
                name.encode("utf-8"),
                num_headers,
                headers_arr,
                include_names_arr,
            )
        )

    def __del__(self):
        _nvrtc.nvrtcDestroyProgram(ctypes.byref(self.program))

    def compile(self, options: List[str] = None):
        """Compile the program with the given options."""
        if options is None:
            options = []
        utf8_options = [option.encode("utf-8") for option in options]
        options = (ctypes.c_char_p * len(options))(*utf8_options)
        _check(_nvrtc.nvrtcCompileProgram(self.program, len(options), options))

    def ptx(self) -> str:
        """Return the PTX code as a string."""
        size = ctypes.c_size_t()
        _check(_nvrtc.nvrtcGetPTXSize(self.program, ctypes.byref(size)))
        ptx = ctypes.create_string_buffer(size.value)
        _check(_nvrtc.nvrtcGetPTX(self.program, ptx))
        return ptx.value.decode("utf-8")

    def cubin(self) -> bytes:
        """Return the CUBIN code as bytes."""
        size = ctypes.c_size_t()
        _check(_nvrtc.nvrtcGetCUBINSize(self.program, ctypes.byref(size)))
        cubin = ctypes.create_string_buffer(size.value)
        _check(_nvrtc.nvrtcGetCUBIN(self.program, cubin))
        return cubin.raw

    def log(self) -> str:
        """Return the compilation log as a string."""
        size = ctypes.c_size_t()
        _check(_nvrtc.nvrtcGetProgramLogSize(self.program, ctypes.byref(size)))
        log = ctypes.create_string_buffer(size.value)
        _check(_nvrtc.nvrtcGetProgramLog(self.program, log))
        return log.value.decode("utf-8")
