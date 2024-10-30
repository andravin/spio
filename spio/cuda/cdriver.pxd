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

    CUresult cuInit(unsigned int Flags)
    CUresult cuDeviceGet(CUdevice *device, int ordinal)
    CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
    CUresult cuDevicePrimaryCtxRelease(CUdevice dev)

    CUresult cuStreamCreate(CUstream *pStream, unsigned int Flags)
    CUresult cuStreamDestroy(CUstream hStream)

    CUresult cuGetErrorString(CUresult error, const char **pStr)    

    CUresult cuModuleLoad(CUmodule *module, const char *fname)
    CUresult cuModuleLoadData(CUmodule *module, const void *image)
    CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
    CUresult cuModuleUnload(CUmodule hmod)

    CUresult cuFuncLoad(CUfunction function)
    CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)
    
    CUresult cuCtxSynchronize()
    CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version)
    CUresult cuDriverGetVersion(int *driverVersion)

    CUresult cuPointerGetAttribute ( void* data, CUpointer_attribute attribute, CUdeviceptr ptr )