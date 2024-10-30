extern "C"
{
    /// A simple add kernel for testing kernel compilation, loading, and execution.
    __global__ void add(const float *x1, const float *x2, float *y)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        y[tid] = x1[tid] + x2[tid];
    }
}
