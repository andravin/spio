"""Classes for CUDA driver API using Cython."""
from posix.dlfcn cimport dlopen, dlsym, dlerror, RTLD_NOW, RTLD_GLOBAL

from spio.cuda cimport cdriver
from cpython.bytes cimport PyBytes_FromString

from dataclasses import dataclass
from typing import Tuple


cdef void* cuda_lib = NULL
cdef bint driver_loaded = False


cdef struct _DriverFns:
    cdriver.cuInit_fn cuInit
    cdriver.cuDeviceGet_fn cuDeviceGet
    cdriver.cuDeviceGetAttribute_fn cuDeviceGetAttribute
    cdriver.cuDeviceGetName_fn cuDeviceGetName
    cdriver.cuDevicePrimaryCtxRetain_fn cuDevicePrimaryCtxRetain
    cdriver.cuDevicePrimaryCtxRelease_fn cuDevicePrimaryCtxRelease
    cdriver.cuStreamCreate_fn cuStreamCreate
    cdriver.cuStreamDestroy_fn cuStreamDestroy

    cdriver.cuGetErrorString_fn cuGetErrorString
    cdriver.cuModuleLoad_fn cuModuleLoad
    cdriver.cuModuleLoadData_fn cuModuleLoadData
    cdriver.cuModuleGetFunction_fn cuModuleGetFunction
    cdriver.cuModuleUnload_fn cuModuleUnload
    cdriver.cuFuncLoad_fn cuFuncLoad
    cdriver.cuFuncSetAttribute_fn cuFuncSetAttribute
    cdriver.cuFuncGetAttribute_fn cuFuncGetAttribute
    cdriver.cuLaunchKernel_fn cuLaunchKernel
    cdriver.cuCtxSynchronize_fn cuCtxSynchronize
    cdriver.cuCtxGetApiVersion_fn cuCtxGetApiVersion
    cdriver.cuDriverGetVersion_fn cuDriverGetVersion

    cdriver.cuPointerGetAttribute_fn cuPointerGetAttribute


cdef _DriverFns _driver_fns


cdef void* _load_symbol(void* handle, const char* name):
    cdef const char* err
    cdef void* sym = dlsym(handle, name)
    if sym == NULL:
        err = dlerror()
        raise RuntimeError(
            f"Could not load symbol {name.decode('ascii')}: "
            f"{PyBytes_FromString(err).decode('utf-8') if err != NULL else 'unknown reason'}")
    return sym


cdef void _load_cuda_driver():
    global cuda_lib, _driver_fns, driver_loaded

    cdef const char* err = NULL

    if driver_loaded:
        return

    # Load the driver library
    for name in [b"libcuda.so.1", b"libcuda.so"]:
        cuda_lib = dlopen(name, RTLD_NOW | RTLD_GLOBAL)
        if cuda_lib != NULL:
            break

    if cuda_lib == NULL:
        raise RuntimeError("Could not load libcuda.so.1 or libcuda.so")

    # Resolve symbols
    _driver_fns.cuInit = <cdriver.cuInit_fn>_load_symbol(cuda_lib, b"cuInit")
    _driver_fns.cuDeviceGet = <cdriver.cuDeviceGet_fn>_load_symbol(cuda_lib, b"cuDeviceGet")
    _driver_fns.cuDeviceGetAttribute = <cdriver.cuDeviceGetAttribute_fn>_load_symbol(cuda_lib, b"cuDeviceGetAttribute")
    _driver_fns.cuDeviceGetName = <cdriver.cuDeviceGetName_fn>_load_symbol(cuda_lib, b"cuDeviceGetName")
    _driver_fns.cuDevicePrimaryCtxRetain = <cdriver.cuDevicePrimaryCtxRetain_fn>_load_symbol(cuda_lib, b"cuDevicePrimaryCtxRetain")
    _driver_fns.cuDevicePrimaryCtxRelease = <cdriver.cuDevicePrimaryCtxRelease_fn>_load_symbol(cuda_lib, b"cuDevicePrimaryCtxRelease")
    _driver_fns.cuStreamCreate = <cdriver.cuStreamCreate_fn>_load_symbol(cuda_lib, b"cuStreamCreate")
    _driver_fns.cuStreamDestroy = <cdriver.cuStreamDestroy_fn>_load_symbol(cuda_lib, b"cuStreamDestroy")
    _driver_fns.cuGetErrorString = <cdriver.cuGetErrorString_fn>_load_symbol(cuda_lib, b"cuGetErrorString")
    _driver_fns.cuModuleLoad = <cdriver.cuModuleLoad_fn>_load_symbol(cuda_lib, b"cuModuleLoad")
    _driver_fns.cuModuleLoadData = <cdriver.cuModuleLoadData_fn>_load_symbol(cuda_lib, b"cuModuleLoadData")
    _driver_fns.cuModuleGetFunction = <cdriver.cuModuleGetFunction_fn>_load_symbol(cuda_lib, b"cuModuleGetFunction")
    _driver_fns.cuModuleUnload = <cdriver.cuModuleUnload_fn>_load_symbol(cuda_lib, b"cuModuleUnload")
    _driver_fns.cuFuncLoad = <cdriver.cuFuncLoad_fn>_load_symbol(cuda_lib, b"cuFuncLoad")
    _driver_fns.cuFuncSetAttribute = <cdriver.cuFuncSetAttribute_fn>_load_symbol(cuda_lib, b"cuFuncSetAttribute")
    _driver_fns.cuFuncGetAttribute = <cdriver.cuFuncGetAttribute_fn>_load_symbol(cuda_lib, b"cuFuncGetAttribute")
    _driver_fns.cuLaunchKernel = <cdriver.cuLaunchKernel_fn>_load_symbol(cuda_lib, b"cuLaunchKernel")
    _driver_fns.cuCtxSynchronize = <cdriver.cuCtxSynchronize_fn>_load_symbol(cuda_lib, b"cuCtxSynchronize")
    _driver_fns.cuCtxGetApiVersion = <cdriver.cuCtxGetApiVersion_fn>_load_symbol(cuda_lib, b"cuCtxGetApiVersion")
    _driver_fns.cuDriverGetVersion = <cdriver.cuDriverGetVersion_fn>_load_symbol(cuda_lib, b"cuDriverGetVersion")
    _driver_fns.cuPointerGetAttribute = <cdriver.cuPointerGetAttribute_fn>_load_symbol(cuda_lib, b"cuPointerGetAttribute")

    driver_loaded = True


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
        _driver_fns.cuGetErrorString(status, &err_str)
        py_err_str = PyBytes_FromString(err_str).decode('utf-8')
        raise ValueError(f"CUDA error: " + py_err_str)


cdef class Function:
    """CUDA kernel function wrapper."""
    cdef cdriver.CUfunction _c_function

    def __cinit__(self):
        _load_cuda_driver()
        self._c_function = NULL

    cdef set_c_function(self, cdriver.CUfunction c_function):
        self._c_function = c_function

    cdef set_max_dynamic_shared_memory_size(self, size):
        """Set the maximum dynamic shared memory size for this function."""
        _check(_driver_fns.cuFuncSetAttribute(self._c_function, cdriver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, size))

    cdef get_max_dynamic_shared_memory_size(self):
        cdef int size
        _check(_driver_fns.cuFuncGetAttribute(&size, cdriver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self._c_function))
        return size

    cdef set_preferred_shared_memory_carveout(self, percentage):
        """Set the preferred shared memory carveout for this function."""
        _check(_driver_fns.cuFuncSetAttribute(
            self._c_function, cdriver.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, percentage))

    cdef get_preferred_shared_memory_carveout(self):
        cdef int percentage
        _check(_driver_fns.cuFuncGetAttribute(
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

        if len(args) > 16:
            raise ValueError("Maximum of 16 arguments supported")

        for idx, arg in enumerate(args):
            if hasattr(arg, '__cuda_array_interface__'):
                data_ptr = arg.__cuda_array_interface__['data'][0]
                if data_ptr != 0:
                    _check(_driver_fns.cuPointerGetAttribute(&arg_ptrs[idx], cdriver.CU_POINTER_ATTRIBUTE_DEVICE_POINTER, data_ptr))
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
        _check(_driver_fns.cuLaunchKernel(
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
        _load_cuda_driver()
        self._c_module = NULL

    def __del__(self):
        self.unload()

    def load(self, fname):
        """Load a CUDA module from a file."""
        _check(_driver_fns.cuModuleLoad(&self._c_module, fname.encode('utf-8')))

    def unload(self):
        """Unload the CUDA module."""
        if driver_loaded and self._c_module is not NULL:
            _check(_driver_fns.cuModuleUnload(self._c_module))
            self._c_module = NULL

    def load_data(self, image):
        """Load a CUDA module from binary data."""
        cdef char *c_image = image
        _check(_driver_fns.cuModuleLoadData(&self._c_module, c_image))

    def get_function(self, name):
        """Get a function from the CUDA module."""
        cdef cdriver.CUfunction _c_function
        _check(_driver_fns.cuModuleGetFunction(&_c_function, self._c_module, name.encode('utf-8')))
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
        _load_cuda_driver()
        _check(_driver_fns.cuDeviceGet(&self._c_device, device_ordinal))
        _check(_driver_fns.cuDevicePrimaryCtxRetain(&self._c_context, self._c_device))

    def set_device(self, device_ordinal):
        cdef cdriver.CUdevice new_device
        _check(_driver_fns.cuDeviceGet(&new_device, device_ordinal))
        if new_device != self._c_device:
            _check(_driver_fns.cuDevicePrimaryCtxRelease(self._c_device))
            self._c_device = new_device
            _check(_driver_fns.cuDevicePrimaryCtxRetain(&self._c_context, self._c_device))

    def get_api_version(self):
        cdef unsigned int version
        _check(_driver_fns.cuCtxGetApiVersion(self._c_context, &version))
        return version

    def __del__(self):
        _driver_fns.cuDevicePrimaryCtxRelease(self._c_device)


def init():
    """Initialize the CUDA driver API."""
    _load_cuda_driver()
    if _driver_fns.cuInit == NULL:
        raise RuntimeError("cuInit symbol was not resolved")
    _check(_driver_fns.cuInit(0))


def ctx_synchronize():
    """Synchronize the current CUDA context."""
    _load_cuda_driver()
    _check(_driver_fns.cuCtxSynchronize())


def get_ctx_api_version():
    """Get the CUDA context API version."""
    cdef unsigned int version
    _load_cuda_driver()
    _check(_driver_fns.cuCtxGetApiVersion(NULL, &version))
    return version


def get_driver_version():
    """Get the CUDA driver version."""
    cdef int version
    _load_cuda_driver()
    _check(_driver_fns.cuDriverGetVersion(&version))
    return version


def get_multiprocessor_count(device_ordinal=0):
    """Get the number of multiprocessors on the given device."""
    cdef int count
    cdef cdriver.CUdevice device
    _load_cuda_driver()
    _check(_driver_fns.cuDeviceGet(&device, device_ordinal))
    _check(_driver_fns.cuDeviceGetAttribute(
        &count, cdriver.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))
    return count


def get_l2_cache_size(device_ordinal=0):
    """Get the size of the L2 cache on the given device."""
    cdef int size
    cdef cdriver.CUdevice device
    _load_cuda_driver()
    _check(_driver_fns.cuDeviceGet(&device, device_ordinal))
    _check(_driver_fns.cuDeviceGetAttribute(
        &size, cdriver.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device))
    return size


def get_device_name(device_ordinal=0):
    """Get the name of the given device."""
    cdef char name[256]
    cdef cdriver.CUdevice device
    _load_cuda_driver()
    _check(_driver_fns.cuDeviceGet(&device, device_ordinal))
    _check(_driver_fns.cuDeviceGetName(name, 256, device))
    return name.decode('utf-8')


def get_compute_capability(device_ordinal=0):
    """Return the compute capability of the given device as a tuple (major, minor)."""
    cdef int major, minor
    cdef cdriver.CUdevice device
    _load_cuda_driver()
    _check(_driver_fns.cuDeviceGet(&device, device_ordinal))
    _check(_driver_fns.cuDeviceGetAttribute(&major, cdriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))
    _check(_driver_fns.cuDeviceGetAttribute(&minor, cdriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))
    return (major, minor)


def get_max_shared_memory_per_block_optin(device_ordinal=0):
    """Get the maximum shared memory per block (opt-in) on the given device."""
    cdef int size
    cdef cdriver.CUdevice device
    _load_cuda_driver()
    _check(_driver_fns.cuDeviceGet(&device, device_ordinal))
    _check(_driver_fns.cuDeviceGetAttribute(
        &size, cdriver.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device))
    return size


def get_device_attributes(device_ordinal=0):
    """Return a dataclass with the device attributes."""
    _load_cuda_driver()
    return DeviceAttributes(
        name=get_device_name(device_ordinal),
        multiprocessor_count=get_multiprocessor_count(device_ordinal),
        l2_cache_size=get_l2_cache_size(device_ordinal),
        compute_capability=get_compute_capability(device_ordinal),
        max_shared_memory_per_block_optin=get_max_shared_memory_per_block_optin(device_ordinal)
    )
