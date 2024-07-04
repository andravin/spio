#include <cuda_pipeline.h>

extern "C"
{
    __device__ constexpr int cdiv(int n, int d)
    {
        return (n + d - 1) / d;
    }

    // Pipeline Primitive Interface:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pipeline-interface
    __global__ void conv_group_4_32w_4h_64c_test(
        uint4 *__restrict__ out,
        const uint4 *__restrict__ in,
        const uint4 *__restrict__ weights)
    {
        constexpr int C = 32;
        constexpr int C8 = C / 8;
        constexpr int GROUP_WIDTH = 4;

        constexpr int P = 4;
        constexpr int Q = 32;

        constexpr int PADDING = 1;

        constexpr int WARPS = 8;
        constexpr int THREADS = WARPS * 32;

        constexpr int BLOCK_H = P + 2;
        constexpr int BLOCK_W = Q + 2;
        constexpr int BLOCK_C8 = WARPS;
        constexpr int UNIT = 2;

        constexpr int LOAD_VECTOR_SIZE = 16;
        constexpr int NUM_BYTES_PER_BLOCK_LOAD = THREADS * LOAD_VECTOR_SIZE;

        constexpr int NUM_INPUT_BYTES_PER_ROW = BLOCK_W * BLOCK_C8 * (8 * UNIT);

        constexpr int NUM_BLOCK_LOADS = cdiv(NUM_INPUT_BYTES_PER_ROW, NUM_BYTES_PER_BLOCK_LOAD);

        constexpr int NUM_THREAD_LOADS = cdiv(NUM_INPUT_BYTES_PER_ROW, LOAD_VECTOR_SIZE);

        __shared__ uint4 smem_in[BLOCK_H * BLOCK_W * C8];
        __shared__ uint4 smem_weights[9 * C8 * GROUP_WIDTH];

        int tid = threadIdx.x;

        // Load the inputs to shared memory.
        int thread_load_idx = tid;
        const uint4 *in_load[NUM_BLOCK_LOADS];
        int in_load_size[NUM_BLOCK_LOADS];
        bool do_load[NUM_BLOCK_LOADS];
        uint4 *in_smem_store[NUM_BLOCK_LOADS];
        for (int block_load_idx = 0; block_load_idx < NUM_BLOCK_LOADS; ++block_load_idx)
        {
            int thread_load_idx = block_load_idx * THREADS + tid;
            int thread_c8 = thread_load_idx % C8;
            int thread_x = ((thread_load_idx / C8) % BLOCK_W) - PADDING;
            bool x_inbounds = thread_x >= 0 && thread_x < Q;
            do_load[block_load_idx] = thread_load_idx < NUM_THREAD_LOADS;
            in_load_size[block_load_idx] = x_inbounds ? LOAD_VECTOR_SIZE : 0;
            in_load[block_load_idx] = &in[thread_x * C8 + thread_c8];
            in_smem_store[block_load_idx] = &smem_in[(thread_x + PADDING) * C8 + thread_c8];
        }

        for (int y = 0; y < BLOCK_H; ++y)
        {
            for (int block_load_idx = 0; block_load_idx < NUM_BLOCK_LOADS; ++block_load_idx)
            {
                if (do_load[block_load_idx])
                {
                    __pipeline_memcpy_async(
                        in_smem_store[block_load_idx] + y * BLOCK_W * C8,
                        in_load[block_load_idx] + y * Q * C8,
                        LOAD_VECTOR_SIZE,
                        in_load_size[block_load_idx]);
                }
            }
            __pipeline_commit();
        }

    
        __syncthreads();
        __pipeline_wait_prior(0);

        // XXX Use the shared memory to avoid compiler warnings.
        smem_weights[0] = smem_in[0];
        out[0] = smem_weights[0];
        out[1] = smem_in[0];

        // Load weights to shared memory.

        // Load weights to registers.

        // For each strip of input rows ..

        // .. load the strip of input rows to share memory.

        // .. for each horizontal shift of the input rows

        //     ..  for each input row in the strip

        //          .. load the shift of the input row to registers.

        //          .. multiply the row by the corresponding column of the weights tensor

        //          .. accumulate onto the appropriate outputs.

        //     .. store any completed outputs to global memory.
    }
}
