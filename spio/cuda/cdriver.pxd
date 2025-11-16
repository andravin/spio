cdef extern from "cuda.h":
    ctypedef enum CUresult:
        CUDA_SUCCESS = 0
        CUDA_ERROR_INVALID_VALUE
        CUDA_ERROR_OUT_OF_MEMORY
        CUDA_ERROR_NOT_INITIALIZED
        CUDA_ERROR_DEINITIALIZED
        CUDA_ERROR_PROFILER_DISABLED
        CUDA_ERROR_PROFILER_NOT_INITIALIZED
        CUDA_ERROR_PROFILER_ALREADY_STARTED
        CUDA_ERROR_PROFILER_ALREADY_STOPPED       
        CUDA_ERROR_NO_DEVICE = 100
        CUDA_ERROR_INVALID_IMAGE = 200
        CUDA_ERROR_INVALID_CONTEXT
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT
        CUDA_ERROR_ILLEGAL_ADDRESS = 700
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
        CUDA_ERROR_LAUNCH_TIMEOUT = 702
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
        CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
        CUDA_ERROR_ASSERT = 710
        CUDA_ERROR_TOO_MANY_PEERS = 711
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
        CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
        CUDA_ERROR_HARDWARE_STACK_ERROR = 714
        CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
        CUDA_ERROR_MISALIGNED_ADDRESS = 716
        CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
        CUDA_ERROR_INVALID_PC = 718
        CUDA_ERROR_LAUNCH_FAILED = 719
        CUDA_ERROR_NOT_PERMITTED = 800
        CUDA_ERROR_NOT_SUPPORTED = 801
        CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 802
        CUDA_ERROR_UNKNOWN = 999

    ctypedef enum CUpointer_attribute:
        CU_POINTER_ATTRIBUTE_CONTEXT = 1
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
        CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
        CU_POINTER_ATTRIBUTE_HOST_POINTER = 4
        CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5
        CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6
        CU_POINTER_ATTRIBUTE_BUFFER_ID = 7

    ctypedef enum CUdevice_attribute:
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
        CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97

    ctypedef enum CUfunction_attribute:
        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
        CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
        CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
        CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
        CU_FUNC_ATTRIBUTE_NUM_REGS = 4
        CU_FUNC_ATTRIBUTE_PTX_VERSION = 5
        CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6
        CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9



    ctypedef enum CUstream_flags:
        CU_STREAM_DEFAULT = 0
        CU_STREAM_NON_BLOCKING = 1

    cdef struct CUctx_st:
        pass
    
    cdef struct CUmod_st:
        pass

    cdef struct CUfunction_st:
        pass

    cdef struct CUstream_st:
        pass

    ctypedef CUctx_st* CUcontext
    ctypedef int CUdevice_v1
    ctypedef CUdevice_v1 CUdevice
    ctypedef unsigned int CUdeviceptr_v2

    ctypedef CUfunction_st* CUfunction
    ctypedef CUmod_st* CUmodule
    ctypedef CUstream_st* CUstream
    ctypedef CUdeviceptr_v2 CUdeviceptr

ctypedef CUresult (*cuInit_fn)(unsigned int Flags)
ctypedef CUresult (*cuDeviceGet_fn)(CUdevice *device, int ordinal)
ctypedef CUresult (*cuDeviceGetAttribute_fn)(int *pi, int attrib, CUdevice dev)
ctypedef CUresult (*cuDeviceGetName_fn)(char* name, int  len, CUdevice dev)
ctypedef CUresult (*cuDevicePrimaryCtxRetain_fn)(CUcontext *pctx, CUdevice dev)
ctypedef CUresult (*cuDevicePrimaryCtxRelease_fn)(CUdevice dev)

ctypedef CUresult (*cuStreamCreate_fn)(CUstream *pStream, unsigned int Flags)
ctypedef CUresult (*cuStreamDestroy_fn)(CUstream hStream)
ctypedef CUresult (*cuGetErrorString_fn)(CUresult error, const char **pStr)

ctypedef CUresult (*cuModuleLoad_fn)(CUmodule *module, const char *fname)
ctypedef CUresult (*cuModuleLoadData_fn)(CUmodule *module, const void *image)
ctypedef CUresult (*cuModuleGetFunction_fn)(CUfunction *hfunc, CUmodule hmod, const char *name)
ctypedef CUresult (*cuModuleUnload_fn)(CUmodule hmod)

ctypedef CUresult (*cuFuncLoad_fn)(CUfunction function)
ctypedef CUresult (*cuFuncSetAttribute_fn)(CUfunction function, int attrib, int value)
ctypedef CUresult (*cuFuncGetAttribute_fn)(int *pi, int attrib, CUfunction function)
ctypedef CUresult (*cuLaunchKernel_fn)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)

ctypedef CUresult (*cuCtxSynchronize_fn)()
ctypedef CUresult (*cuCtxGetApiVersion_fn)(CUcontext ctx, unsigned int *version)
ctypedef CUresult (*cuDriverGetVersion_fn)(int *driverVersion)

ctypedef CUresult (*cuPointerGetAttribute_fn)(void* data, CUpointer_attribute attribute, CUdeviceptr ptr)