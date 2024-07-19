#include "spio/ldmatrix.cuh"

using namespace spio;

extern "C"
{
    /// @brief  Test the ldmatrix_x1 function.
    /// @details The functions loads a tensor core matrix fragment from shared memory to registers.
    /// @param out The output matrix fragment in the order as stored in registers.
    /// @param in The input 8x8 matrix with float16 elements.
    /// @return
    __global__ void ldmatrix_x1(
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

    __global__ void ldmatrix_x2(
        unsigned *__restrict__ out,
        const unsigned *__restrict__ in)
    {
        __shared__ unsigned smem[32 * 2];

        int lane = threadIdx.x % 32;

        for (int idx = lane; idx < 32 * 2; idx += 32)
        {
            smem[idx] = in[idx];
        }
        __syncthreads();

        int fragment_idx = (lane / 8) % 2;
        int row_idx = lane % 8;
        uint2 a = ldmatrix_x2(smem + row_idx * 4 + fragment_idx * 32);

        out[lane] = a.x;
        out[lane + 32] = a.y;
    }

    __global__ void ldmatrix_x4(
        unsigned *__restrict__ out,
        const unsigned *__restrict__ in)
    {
        __shared__ unsigned smem[32 * 4];

        int lane = threadIdx.x % 32;

        for (int idx = lane; idx < 32 * 4; idx += 32)
        {
            smem[idx] = in[idx];
        }
        __syncthreads();

        int fragment_idx = (lane / 8) % 4;
        int row_idx = lane % 8;
        uint4 a = ldmatrix_x4(smem + row_idx * 4 + fragment_idx * 32);

        out[lane] = a.x;
        out[lane + 32] = a.y;
        out[lane + 64] = a.z;
        out[lane + 96] = a.w;
    }

    __global__ void ldmatrix_x1_trans(
        unsigned *__restrict__ out,
        const unsigned *__restrict__ in)
    {
        __shared__ unsigned smem[32];

        int lane = threadIdx.x % 32;

        smem[threadIdx.x] = in[lane];
        __syncthreads();

        unsigned a = ldmatrix_x1_trans(smem + (lane % 8) * 4);

        out[lane] = a;
    }

    __global__ void ldmatrix_x2_trans(
        unsigned *__restrict__ out,
        const unsigned *__restrict__ in)
    {
        __shared__ unsigned smem[32 * 2];

        int lane = threadIdx.x % 32;

        for (int idx = lane; idx < 32 * 2; idx += 32)
        {
            smem[idx] = in[idx];
        }
        __syncthreads();

        int fragment_idx = (lane / 8) % 2;
        int row_idx = lane % 8;
        uint2 a = ldmatrix_x2_trans(smem + row_idx * 4 + fragment_idx * 32);

        out[lane] = a.x;
        out[lane + 32] = a.y;
    }

    __global__ void ldmatrix_x4_trans(
        unsigned *__restrict__ out,
        const unsigned *__restrict__ in)
    {
        __shared__ unsigned smem[32 * 4];

        int lane = threadIdx.x % 32;

        for (int idx = lane; idx < 32 * 4; idx += 32)
        {
            smem[idx] = in[idx];
        }
        __syncthreads();

        int fragment_idx = (lane / 8) % 4;
        int row_idx = lane % 8;
        uint4 a = ldmatrix_x4_trans(smem + row_idx * 4 + fragment_idx * 32);

        out[lane] = a.x;
        out[lane + 32] = a.y;
        out[lane + 64] = a.z;
        out[lane + 96] = a.w;
    }
}