"""Classes for CUDA driver API using Cython."""
from spio.cuda cimport cdriver
from cpython.bytes cimport PyBytes_FromString

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DeviceAttributes:
    """Attributes of a CUDA device."""

    multiprocessor_count: int
    l2_cache_size: int
    name: str = None
    compute_capability: Tuple[int, int] = None
    max_shared_memory_per_block_optin: int = None
    max_shared_memory_per_block: int = 48 * 1024
    num_partitions_per_sm: int = 4


@dataclass(frozen=True)
class FunctionAttributes:
    """Attributes of a CUDA function."""
    max_dynamic_shared_memory_size: int = None
    preferred_shared_memory_carveout: int = None


cdef _check(cdriver.CUresult status):
    cdef const char *err_str
    if status != cdriver.CUDA_SUCCESS:
        cdriver.cuGetErrorString(status, &err_str)
        py_err_str = PyBytes_FromString(err_str).decode('utf-8')
        raise ValueError(f"CUDA error: " + py_err_str)

cdef class Function:
    """CUDA kernel function wrapper."""
    cdef cdriver.CUfunction _c_function

    def __cinit__(self):
        self._c_function = NULL

    cdef set_c_function(self, cdriver.CUfunction c_function):
        self._c_function = c_function

    cdef set_max_dynamic_shared_memory_size(self, size):
        """Set the maximum dynamic shared memory size for this function."""
        _check(cdriver.cuFuncSetAttribute(self._c_function, cdriver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, size))

    cdef get_max_dynamic_shared_memory_size(self):
        cdef int size
        _check(cdriver.cuFuncGetAttribute(&size, cdriver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self._c_function))
        return size

    cdef set_preferred_shared_memory_carveout(self, percentage):
        """Set the preferred shared memory carveout for this function."""
        _check(cdriver.cuFuncSetAttribute(
            self._c_function, cdriver.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, percentage))

    cdef get_preferred_shared_memory_carveout(self):
        cdef int percentage
        _check(cdriver.cuFuncGetAttribute(
            &percentage, cdriver.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, self._c_function))
        return percentage

    def get_attributes(self):
        """Get attributes of the CUDA function."""
        return FunctionAttributes(
            max_dynamic_shared_memory_size=self.get_max_dynamic_shared_memory_size(),
            preferred_shared_memory_carveout=self.get_preferred_shared_memory_carveout()
        )

    def set_attributes(self, attr: FunctionAttributes):
        """Set attributes of the CUDA function."""
        if attr.preferred_shared_memory_carveout is not None:
            self.set_preferred_shared_memory_carveout(attr.preferred_shared_memory_carveout)
        if attr.max_dynamic_shared_memory_size is not None:
            self.set_max_dynamic_shared_memory_size(attr.max_dynamic_shared_memory_size)

    def launch(self, grid, block, args, shared_mem_bytes=0):
        """Launch the CUDA kernel function."""
        cdef cdriver.CUdeviceptr arg_ptrs[16]
        cdef long long arg_ints[16]
        cdef float arg_floats[16]
        cdef void *kernel_params[16]

        for idx, arg in enumerate(args):
            if hasattr(arg, '__cuda_array_interface__'):
                data_ptr = arg.__cuda_array_interface__['data'][0]
                if data_ptr != 0:
                    _check(cdriver.cuPointerGetAttribute(&arg_ptrs[idx], cdriver.CU_POINTER_ATTRIBUTE_DEVICE_POINTER, data_ptr))
                else:
                    arg_ptrs[idx] = 0
                kernel_params[idx] = &arg_ptrs[idx]
            elif arg is None:
                arg_ptrs[idx] = 0
                kernel_params[idx] = &arg_ptrs[idx]
            elif isinstance(arg, int):
                arg_ints[idx] = arg
                kernel_params[idx] = &arg_ints[idx]
            elif isinstance(arg, float):
                arg_floats[idx] = arg
                kernel_params[idx] = &arg_floats[idx]
            else:
                raise ValueError(f"Unsupported argument type: {type(arg)}")
        _check(cdriver.cuLaunchKernel(
            self._c_function,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem_bytes,
            NULL, # stream
            kernel_params,
            NULL # extra
        ))

cdef class Module:
    """CUDA module wrapper."""
    cdef cdriver.CUmodule _c_module

    def __cinit__(self):
        self._c_module = NULL

    def __del__(self):
        self.unload()

    def load(self, fname):
        """Load a CUDA module from a file."""
        _check(cdriver.cuModuleLoad(&self._c_module, fname.encode('utf-8')))

    def unload(self):
        """Unload the CUDA module."""
        if self._c_module is not NULL:
            _check(cdriver.cuModuleUnload(self._c_module))
            self._c_module = NULL

    def load_data(self, image):
        """Load a CUDA module from binary data."""
        cdef char *c_image = image
        _check(cdriver.cuModuleLoadData(&self._c_module, c_image))

    def get_function(self, name):
        """Get a function from the CUDA module."""
        cdef cdriver.CUfunction _c_function
        _check(cdriver.cuModuleGetFunction(&_c_function, self._c_module, name.encode('utf-8')))
        f = Function()
        f.set_c_function(_c_function)
        return f


cdef class PrimaryContextGuard:
    """CUDA primary context guard.
    
    This class gets and retains the primary context for a given device.
    It releases the context when the object is deleted.
    """
    cdef cdriver.CUcontext _c_context
    cdef cdriver.CUdevice _c_device

    def __cinit__(self, device_ordinal=0):
        _check(cdriver.cuDeviceGet(&self._c_device, device_ordinal))
        _check(cdriver.cuDevicePrimaryCtxRetain(&self._c_context, self._c_device))

    def set_device(self, device_ordinal):
        cdef cdriver.CUdevice new_device
        _check(cdriver.cuDeviceGet(&new_device, device_ordinal))
        if new_device != self._c_device:
            _check(cdriver.cuDevicePrimaryCtxRelease(self._c_device))
            self._c_device = new_device
            _check(cdriver.cuDevicePrimaryCtxRetain(&self._c_context, self._c_device))

    def get_api_version(self):
        cdef unsigned int version
        _check(cdriver.cuCtxGetApiVersion(self._c_context, &version))
        return version

    def __del__(self):
        cdriver.cuDevicePrimaryCtxRelease(self._c_device)


def init():
    """Initialize the CUDA driver API."""
    _check(cdriver.cuInit(0))


def ctx_synchronize():
    """Synchronize the current CUDA context."""
    _check(cdriver.cuCtxSynchronize())


def get_ctx_api_version():
    """Get the CUDA context API version."""
    cdef unsigned int version
    _check(cdriver.cuCtxGetApiVersion(NULL, &version))
    return version


def get_driver_version():
    """Get the CUDA driver version."""
    cdef int version
    _check(cdriver.cuDriverGetVersion(&version))
    return version


def get_multiprocessor_count(device_ordinal=0):
    """Get the number of multiprocessors on the given device."""
    cdef int count
    cdef cdriver.CUdevice device
    _check(cdriver.cuDeviceGet(&device, device_ordinal))
    _check(cdriver.cuDeviceGetAttribute(
        &count, cdriver.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))
    return count


def get_l2_cache_size(device_ordinal=0):
    """Get the size of the L2 cache on the given device."""
    cdef int size
    cdef cdriver.CUdevice device
    _check(cdriver.cuDeviceGet(&device, device_ordinal))
    _check(cdriver.cuDeviceGetAttribute(
        &size, cdriver.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device))
    return size


def get_device_name(device_ordinal=0):
    """Get the name of the given device."""
    cdef char name[256]
    cdef cdriver.CUdevice device
    _check(cdriver.cuDeviceGet(&device, device_ordinal))
    _check(cdriver.cuDeviceGetName(name, 256, device))
    return name.decode('utf-8')


def get_compute_capability(device_ordinal=0):
    """Return the compute capability of the given device as a tuple (major, minor)."""
    cdef int major, minor
    cdef cdriver.CUdevice device
    _check(cdriver.cuDeviceGet(&device, device_ordinal))
    _check(cdriver.cuDeviceGetAttribute(&major, cdriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))
    _check(cdriver.cuDeviceGetAttribute(&minor, cdriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))
    return (major, minor)


def get_max_shared_memory_per_block_optin(device_ordinal=0):
    """Get the maximum shared memory per block (opt-in) on the given device."""
    cdef int size
    cdef cdriver.CUdevice device
    _check(cdriver.cuDeviceGet(&device, device_ordinal))
    _check(cdriver.cuDeviceGetAttribute(
        &size, cdriver.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device))
    return size


def get_device_attributes(device_ordinal=0):
    """Return a dataclass with the device attributes."""
    return DeviceAttributes(
        name=get_device_name(device_ordinal),
        multiprocessor_count=get_multiprocessor_count(device_ordinal),
        l2_cache_size=get_l2_cache_size(device_ordinal),
        compute_capability=get_compute_capability(device_ordinal),
        max_shared_memory_per_block_optin=get_max_shared_memory_per_block_optin(device_ordinal)
    )
