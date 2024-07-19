#include <cuda_pipeline.h>

#include "my_params.h"

using namespace MyParams;

extern "C"
{
    __global__ void memcpy_simple(
        uint4 *__restrict__ dst,
        const uint4 *__restrict__ src)
    {
        __shared__ uint4 smem[2 * THREADS];

        int block_x4 = blockIdx.x * BLOCK_X4;

        uint4 * dst_store = dst + block_x4 + threadIdx.x;
        const uint4 *src_load = src + block_x4 + threadIdx.x;

        uint4 *smem_write = smem + threadIdx.x;
        const uint4 *smem_read = smem + threadIdx.x;

        int ping_pong = 0;

        __pipeline_memcpy_async(smem_write, src_load, sizeof(uint4), 0);
        __pipeline_commit();

        ping_pong = 1 - ping_pong;

        for (int i = 0; i < ITERS; ++i)
        {
            if ((i + 1) < ITERS)
            {
                __pipeline_memcpy_async(
                    smem_write + ping_pong * THREADS,
                    src_load + (i + 1) * THREADS,
                     sizeof(uint4),
                     0);
            }
            __pipeline_commit();

            __pipeline_wait_prior(1);
            __syncthreads();

            ping_pong = 1 - ping_pong;
            dst_store[i * THREADS] = smem_read[ping_pong * THREADS];
            __syncthreads();
        }
        __pipeline_wait_prior(0);
    }
}
