#include <cuda_pipeline.h>
#include <cuda_fp16.h>

#include "spio/mma.cuh"
#include "spio/ldmatrix.cuh"
#include "spio/async_loader.cuh"

// TODO tell code analysis to ignore these generated header files.
// https://devblogs.microsoft.com/cppblog/customized-warning-levels-and-code-analysis-for-external-headers/
#include "my_header.h"

using namespace spio;
using namespace Params;
using namespace Tiles;

extern "C"
{
    // Uses Pipeline Primitive Interface for async memory copy. Referenced in the CUDA Programming Guide.
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pipeline-interface
    __global__ void conv_group_4_16w_4h_64c(
        uint4 *__restrict__ out,
        const uint4 *__restrict__ in,
        const uint4 *__restrict__ weights)
    {
        using InputLoader = AsyncLoader<Input, SmemInput, CHUNK_H, CHUNK_P, THREADS>;

        // Shared memory buffer for chunks of the input tensor.
        __shared__ uint4 smem_in[SmemInput::size];

        // Shared memory buffer used for both weights and chunks of the output tensor.
        __shared__ uint4 smem_buf[_max(ConstSmemWeights::size, ConstSmemOutput::size)];

        // Determine the block coordinates.
        BlockIdx block_idx(blockIdx.x);
        auto block_n = block_idx.n();
        auto block_p = block_idx.p() * BLOCK_P;
        auto block_q = block_idx.q() * BLOCK_Q;
        auto block_c8 = block_idx.c8() * BLOCK_C8;

        // Load weights to shared memory.
        auto block_c = block_c8 * 8;
        auto weights_load = Weights(weights).k(block_c);
        for (int idx = threadIdx.x; idx < ConstSmemWeights::size; idx += THREADS)
        {
            __pipeline_memcpy_async(
                smem_buf + idx,
                weights_load.get() + idx,
                sizeof(ConstSmemWeights::data_type),
                0);
        }
        __pipeline_commit();

        // Load the first chunk of input to shared memory.
        auto block_y = block_p - PADDING;
        auto block_x = block_q - PADDING;
        InputLoader loader(smem_in, in, block_n, block_y, block_x, block_c8);
        loader.first_load();

        // Wait for the weights loads from DRAM -> SMEM to complete.
        __pipeline_wait_prior(1);
        __syncthreads();

        using WeightsFrag = MMA_N8_K8_F16_B;
        WeightsFrag wgts[R * S];

        // Setup tile for weights loading from smem.
        ConstSmemWeights smem_weights_load(smem_buf);
        {
            SmemWeightsLoadIdx smem_weights_load_idx(threadIdx.x);
            auto weights_load_k = smem_weights_load_idx.kd8() * 8 + smem_weights_load_idx.km8();
            smem_weights_load = smem_weights_load.k(weights_load_k);
        }

        // Load weights to registers.
        for (auto rs = 0; rs < R * S; ++rs)
        {
            wgts[rs].vector() = ldmatrix_x1(smem_weights_load.s(rs).get());
        }

        // Setup tile for input loading from smem.
        ConstSmemInput in_smem_load(smem_in);
        {
            SmemInputLoadIdx smem_input_load_idx(threadIdx.x);
            in_smem_load = in_smem_load.x(smem_input_load_idx.x()).c8(smem_input_load_idx.c8());
        }

        // Setup tile for output storing to smem.
        SmemOutput smem_output_store_q0;
        SmemOutput smem_output_store_q8;
        {
            SmemOutputStoreIdx thread_idx(threadIdx.x);
            auto group_idx = thread_idx.warp();
            auto lane_idx = thread_idx.lane();
            auto warp_c2 = group_idx * 4;
            auto lane_c2 = Acc::c2(lane_idx);
            auto lane_q0 = Acc::q(lane_idx, 0);
            auto lane_q8 = Acc::q(lane_idx, 1);
            auto smem_output_store = SmemOutput(reinterpret_cast<__half2 *>(smem_buf)).c2(warp_c2 + lane_c2);
            smem_output_store_q0 = smem_output_store.q(lane_q0);
            smem_output_store_q8 = smem_output_store.q(lane_q8);
        }

        // Setup tile for output storing to global memory.
        auto out_tensor = Output(out).n(block_n);
        ConstSmemOutput smem_out(smem_buf);

        // Iterate over all chunks of the output.
        // A chunk has 16-columns and CHUNK_P rows.
        for (auto chunk_p = 0; chunk_p < BLOCK_P; chunk_p += CHUNK_P)
        {
            // Asynchronously load the input chunk for the next loop iteration.
            // Rely on bounds-checking in the loader class to prevent out-of-bounds loads.
            loader.load(chunk_p + CHUNK_H);

            // Wait for the current input chunk to finish loading.
            __pipeline_wait_prior(1);
            __syncthreads();

            // Initialize the accumulators.
            Acc acc[CHUNK_P];
            for (auto p = 0; p < CHUNK_P; ++p)
            {
                acc[p].zero();
            }

            // Iterate over the columns of the convolution kernel.
            for (auto s = 0; s < S; ++s)
            {
                // Load input rows shifted by s from smem.
                MMA_M16_K8_F16_A in_i[CHUNK_H];
                for (auto i = 0; i < CHUNK_H; ++i)
                {
                    auto y = chunk_p + i;
                    auto y_idx = y % NUM_SMEM_INPUT_ROWS;
                    in_i[i].vector() = ldmatrix_x2(in_smem_load.y(y_idx).x(s).get());
                }

                // Iterate over the rows of the convolution kernel.
                for (auto r = 0; r < R; ++r)
                {
                    // Iterate over the rows of the output chunk.
                    for (auto p = 0; p < CHUNK_P; ++p)
                    {
                        // Compute the row of the input that multiplies row-r of the kernel to update row-p of the chunk.
                        auto i = p + r;

                        // Multiply the input row by the kernel row.
                        // This is a 16q x 8k x 8c matrix multiplication, because the input row has 16q columns and 8c channels;
                        // and the kernel has 8c input channels and 8k output channels.
                        mma_m16_n8_k8(acc[p].vec4(), in_i[i].reg2(), wgts[r * S + s].reg(), acc[p].vec4());
                    }
                }
            }

            // Store the result to smem.
            for (auto p = 0; p < CHUNK_P; ++p)
            {
                // Just as 8c to the inner dimension size? But then reading from
                // the buffer is no longer contiguous.
                *smem_output_store_q0.p(p) = acc[p].to_half2(0);
                *smem_output_store_q8.p(p) = acc[p].to_half2(1);
            }
            __syncthreads();

            // Transfer the result from smem to the output tensor.
            for (int i = threadIdx.x; i < static_cast<int>(SmemOutputLoadIdx::size); i += THREADS)
            {
                SmemOutputLoadIdx thread_idx(i);
                auto thread_c8 = thread_idx.c8();
                auto thread_p = thread_idx.p();
                auto thread_q = thread_idx.q();
                auto c8 = block_c8 + thread_c8;
                auto p = block_p + chunk_p + thread_p;
                auto q = block_q + thread_q;
                if (p < H && q < W && c8 < C8)
                {
                    *out_tensor.p(p).q(q).c8(c8) = *smem_out.p(thread_p).q(thread_q).c8(thread_c8);
                }
            }
        }
        __pipeline_wait_prior(0);
    }
}