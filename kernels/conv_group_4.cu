#include <cuda_pipeline.h>
#include <cuda_fp16.h>

#include "spio/mma.h"
#include "spio/ldmatrix.h"

#include "my_indices.h"
#include "my_tensors.h"
#include "my_params.h"
#include "my_tiles.h"

using namespace spio;

using namespace MyParams;
using namespace MyTiles;

extern "C"
{
    __device__ constexpr int cdiv(int n, int d)
    {
        return (n + d - 1) / d;
    }

    // Pipeline Primitive Interface:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pipeline-interface
    __global__ void conv_group_4_16w_4h_64c_test(
        uint4 *__restrict__ out,
        const uint4 *__restrict__ in,
        const uint4 *__restrict__ weights)
    {
        constexpr int UNIT = 2;

        constexpr int CHUNK_H = CHUNK_P + R - 1;

        constexpr int LOAD_VECTOR_SIZE = 16;
        constexpr int MAX_BYTES_PER_BLOCK_LOAD = THREADS * LOAD_VECTOR_SIZE;

        constexpr int NUM_BYTES_PER_INPUT_ROW = BLOCK_W * BLOCK_C8 * (8 * UNIT);

        constexpr int NUM_BYTES_PER_INPUT_CHUNK = CHUNK_P * NUM_BYTES_PER_INPUT_ROW;
        constexpr int NUM_LOADS_PER_INPUT_CHUNK = cdiv(NUM_BYTES_PER_INPUT_CHUNK, MAX_BYTES_PER_BLOCK_LOAD);

        constexpr int FIRST_CHUNK_P = CHUNK_H;
        constexpr int NUM_BYTES_FIRST_INPUT_CHUNK = FIRST_CHUNK_P * NUM_BYTES_PER_INPUT_ROW;
        constexpr int NUM_LOADS_FIRST_INPUT_CHUNK = cdiv(NUM_BYTES_FIRST_INPUT_CHUNK, MAX_BYTES_PER_BLOCK_LOAD);
        constexpr int NUM_THREAD_LOADS_FIRST_INPUT_CHUNK = cdiv(NUM_BYTES_FIRST_INPUT_CHUNK, LOAD_VECTOR_SIZE);

        // constexpr int NUM_LOADS_PER_INPUT_ROW = cdiv(NUM_BYTES_PER_INPUT_ROW, MAX_BYTES_PER_BLOCK_LOAD);
        // constexpr int NUM_THREAD_LOADS = cdiv(NUM_BYTES_PER_INPUT_ROW, LOAD_VECTOR_SIZE);

        constexpr int NUM_WEIGHTS = C * R * S * GROUP_WIDTH;
        constexpr int NUM_WEIGHTS_VECTORS = NUM_WEIGHTS * UNIT / LOAD_VECTOR_SIZE;

        // constexpr int NUM_PADDED_INPUTS = P * BLOCK_W * C8;

        __shared__ uint4 smem_weights[NUM_WEIGHTS_VECTORS];
        __shared__ uint4 smem_in[NUM_BYTES_FIRST_INPUT_CHUNK / sizeof(uint4)];

        // Load weights to shared memory.
        for (int idx = threadIdx.x; idx < NUM_WEIGHTS_VECTORS; idx += THREADS)
        {
            __pipeline_memcpy_async(
                smem_weights + idx,
                weights + idx,
                LOAD_VECTOR_SIZE,
                0);
        }
        __pipeline_commit();

        // Load the first chunk of input to shared memory.
        Input in_load[NUM_LOADS_FIRST_INPUT_CHUNK];
        int in_zfill_size[NUM_LOADS_FIRST_INPUT_CHUNK];
        bool do_load[NUM_LOADS_FIRST_INPUT_CHUNK];
        SmemInput in_smem_store[NUM_LOADS_FIRST_INPUT_CHUNK];
        for (int block_load_idx = 0; block_load_idx < NUM_LOADS_FIRST_INPUT_CHUNK; ++block_load_idx)
        {
            int thread_load_idx = block_load_idx * THREADS + threadIdx.x;
            int thread_c8 = thread_load_idx % C8;
            int thread_x = ((thread_load_idx / C8) % BLOCK_W) - PADDING;
            int thread_y = thread_load_idx / (BLOCK_W * C8) - PADDING;
            bool xy_inbounds = thread_x >= 0 && thread_x < Q && thread_y >= 0 && thread_y < P;
            do_load[block_load_idx] = thread_load_idx < NUM_THREAD_LOADS_FIRST_INPUT_CHUNK;
            in_zfill_size[block_load_idx] = xy_inbounds ? 0 : LOAD_VECTOR_SIZE;
            in_load[block_load_idx] = Input(in).y(thread_y).x(thread_x).c8(thread_c8);
            in_smem_store[block_load_idx] = SmemInput(smem_in).y(thread_y + PADDING).x(thread_x + PADDING).c8(thread_c8);
        }

        for (int block_load_idx = 0; block_load_idx < NUM_LOADS_FIRST_INPUT_CHUNK; ++block_load_idx)
        {
            if (do_load[block_load_idx])
            {
                __pipeline_memcpy_async(
                    in_smem_store[block_load_idx].get(),
                    in_load[block_load_idx].get(),
                    LOAD_VECTOR_SIZE,
                    in_zfill_size[block_load_idx]);
            }
        }
        __pipeline_commit();

        // Weight for all loads to complete.
        __syncthreads();
        __pipeline_wait_prior(0);

        ThreadIdx thread_idx(threadIdx.x);
        int warp_idx = thread_idx.warp();
        int lane_idx = thread_idx.lane();
        int group_idx = warp_idx;

        using WeightsFrag = MMA_M16_N8_K8_F16_B;
        WeightsFrag wgts[R * S];

        // Load weights to registers.
        int warp_k = group_idx * GROUP_WIDTH;
        int lane_weights_load_k = lane_idx % GROUP_WIDTH;
        int weights_load_k = warp_k + lane_weights_load_k;
        auto smem_weights_load = ConstSmemWeights(smem_weights).k(weights_load_k);
        for (int rs = 0; rs < R * S; ++rs)
        {
            wgts[rs].vector() = ldmatrix_x1(smem_weights_load.s(rs).get());
        }

        // Initialize the accumulators.
        using Acc = MMA_M16_N8_K8_F32_C;
        Acc acc[P];
        for (int p = 0; p < P; ++p)
        {
            acc[p].zero();
        }

        // Compute the grouped convolution.
        int warp_c8 = group_idx;
        for (int s = 0; s < S; ++s)
        {
            // Load input rows shifted by s from smem.
            MMA_M16_N8_K8_F16_A in_i[CHUNK_H];
            int load_j = (lane_idx % 16) + s;
            auto in_smem_load = ConstSmemInput(smem_in).x(load_j).c8(warp_c8);
            for (int i = 0; i < CHUNK_H; ++i)
            {
                in_i[i].vector() = ldmatrix_x2(in_smem_load.y(i).get());
            }

            // Multiply the input rows by the filters.
            for (int r = 0; r < R; ++r)
            {
                for (int p = 0; p < P; ++p)
                {
                    int i = p + r;
                    mma_m16_n8_k8(acc[p], in_i[i], wgts[r * S + s], acc[p]);
                }
            }
        }

        // Store the result.
        Output out_tensor(reinterpret_cast<__half2 *>(out));
        for (int p = 0; p < P; ++p)
        {
            for (int q8 = 0; q8 < 2; ++q8)
            {
                __half2 acc_fp16 = __float22half2_rn(acc[p].fragment(q8));
                int lane_q = Acc::row(lane_idx, q8);
                int lane_c2 = Acc::col(lane_idx) / 2;
                int q = lane_q;
                int c2 = lane_c2 + warp_c8 * 4;
                *out_tensor.p(p).q(q).c2(c2) = acc_fp16;
            }
        }
    }
}
