"""Classes for CUDA driver API using Cython."""
from spio.cuda cimport cdriver
from cpython.bytes cimport PyBytes_FromString

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

    def launch(self, grid, block, args):
        """Launch the CUDA kernel function."""
        cdef cdriver.CUdeviceptr arg_ptrs[16]
        cdef int arg_ints[16]
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
            0, # shared memory size
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
        """Load a CUDA module from file."""
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
