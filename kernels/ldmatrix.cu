#include "spio/mma.h"

using namespace spio;

extern "C"
{
    /// @brief  Test the ldmatrix_x1 function.
    /// @details The functions loads a tensor core matrix fragment from shared memory to registers.
    /// @param out The output matrix fragment in the order as stored in registers.
    /// @param in The input 8x8 matrix with float16 elements.
    /// @return
    __global__ void ldmatrix_test(
        unsigned *__restrict__ out,
        const unsigned *__restrict__ in)
    {
        __shared__ unsigned smem[32];

        int lane = threadIdx.x % 32;

        smem[threadIdx.x] = in[lane];
        __syncthreads();

        unsigned a = ldmatrix_x1(smem + (lane % 8) * 4);

        out[lane] = a;
    }
}