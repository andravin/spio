#include <cuda_pipeline.h>
#include <cuda_fp16.h>

#include "spio/mma.h"
#include "spio/ldmatrix.h"

#include "my_indices.h"
#include "my_tensors.h"

using namespace spio;

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
        uint4 *__restrict__ padded_in,
        uint4 *__restrict__ weights_out,
        const uint4 *__restrict__ in,
        const uint4 *__restrict__ weights)
    {
        constexpr int C = 64;
        constexpr int R = 3;
        constexpr int S = 3;
        constexpr int C8 = C / 8;
        constexpr int GROUP_WIDTH = 8;

        constexpr int P = 4;
        constexpr int Q = 16;

        constexpr int PADDING = 1;

        constexpr int WARPS = 8;
        constexpr int THREADS = WARPS * 32;

        constexpr int BLOCK_W = Q + 2;
        constexpr int BLOCK_C8 = WARPS;
        constexpr int UNIT = 2;

        constexpr int LOAD_VECTOR_SIZE = 16;
        constexpr int NUM_BYTES_PER_BLOCK_LOAD = THREADS * LOAD_VECTOR_SIZE;

        constexpr int NUM_INPUT_BYTES_PER_ROW = BLOCK_W * BLOCK_C8 * (8 * UNIT);

        constexpr int NUM_BLOCK_LOADS = cdiv(NUM_INPUT_BYTES_PER_ROW, NUM_BYTES_PER_BLOCK_LOAD);

        constexpr int NUM_THREAD_LOADS = cdiv(NUM_INPUT_BYTES_PER_ROW, LOAD_VECTOR_SIZE);

        constexpr int NUM_WEIGHTS = C * R * S * GROUP_WIDTH;
        constexpr int NUM_WEIGHTS_VECTORS = NUM_WEIGHTS * UNIT / LOAD_VECTOR_SIZE;

        constexpr int NUM_PADDED_INPUTS = P * BLOCK_W * C8;

        __shared__ uint4 smem_weights[NUM_WEIGHTS_VECTORS];
        __shared__ uint4 smem_in[NUM_PADDED_INPUTS];

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

        // Load the inputs to shared memory, one row at a time.
        const uint4 *in_load[NUM_BLOCK_LOADS];
        int in_zfill_size[NUM_BLOCK_LOADS];
        bool do_load[NUM_BLOCK_LOADS];
        uint4 *in_smem_store[NUM_BLOCK_LOADS];
        for (int block_load_idx = 0; block_load_idx < NUM_BLOCK_LOADS; ++block_load_idx)
        {
            int thread_load_idx = block_load_idx * THREADS + threadIdx.x;
            int thread_c8 = thread_load_idx % C8;
            int thread_x = ((thread_load_idx / C8) % BLOCK_W) - PADDING;
            bool x_inbounds = thread_x >= 0 && thread_x < Q;
            do_load[block_load_idx] = thread_load_idx < NUM_THREAD_LOADS;
            in_zfill_size[block_load_idx] = x_inbounds ? 0 : LOAD_VECTOR_SIZE;
            in_load[block_load_idx] = &in[InputIdx().w(thread_x).c8(thread_c8)];
            in_smem_store[block_load_idx] = &smem_in[SmemInputIdx().x(thread_x + PADDING).c8(thread_c8)];
        }

        for (int y = 0; y < P; ++y)
        {
            for (int block_load_idx = 0; block_load_idx < NUM_BLOCK_LOADS; ++block_load_idx)
            {
                if (do_load[block_load_idx])
                {
                    __pipeline_memcpy_async(
                        in_smem_store[block_load_idx] + SmemInputIdx().y(y),
                        in_load[block_load_idx] + InputIdx().h(y),
                        LOAD_VECTOR_SIZE,
                        in_zfill_size[block_load_idx]);
                }
            }
            __pipeline_commit();
        }

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
        for (int rs = 0; rs < R * S; ++rs)
        {
            wgts[rs].vector() = ldmatrix_x1(&smem_weights[SmemWeightsIdx().k(weights_load_k).s(rs)]);
        }

        // XXX write inputs to output
        for (int idx = threadIdx.x; idx < NUM_PADDED_INPUTS; idx += THREADS)
        {
            padded_in[idx] = smem_in[idx];
        }

        // XXX write weights to weights_output
        int lane_frag_k = WeightsFrag::col(lane_idx);
        int lane_frag_c = WeightsFrag::row(lane_idx);
        int store_k = warp_k + lane_frag_k;
        for (int rs = 0; rs < R * S; ++rs)
        {
            reinterpret_cast<unsigned *>(weights_out)[WeightsOutIdx().k(store_k).rs(rs).c2(lane_frag_c / 2)] = wgts[rs].vector();
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
            int load_j = (lane_idx % 16) + s;
            auto in_smem_load = ConstSmemInput(smem_in).x(load_j).c8(warp_c8);
            for (int i = 0; i < P; ++i)
            {
                MMA_M16_N8_K8_F16_A in_i;
                in_i.vector() = ldmatrix_x2(in_smem_load.y(i).get());
                int p_min = max(i - (R - 1) + (R / 2), 0);
                int p_max = min(i + (R / 2) + 1, P);
                for (int p = p_min; p < p_max; ++p)
                {
                    int r = i - p + (R / 2);
                    mma_m16_n8_k8(acc[p], in_i, wgts[r * S + s], acc[p]);
                }
            }
        }

        // Store the result.
        Output out_tensor(reinterpret_cast<__half2*>(out));
        for (int p = 0; p < P; ++p) {
            for(int q8 = 0; q8 < 2; ++q8) {
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
