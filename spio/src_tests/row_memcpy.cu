#include <cuda_pipeline.h>

#include "spio/pipeline.h"
#include "spio/mathutil.h"

#include "parameters.h"

using namespace spio;

extern "C"
{
    __global__ void row_memcpy(
        float4 *__restrict__ dst,
        const float4 *__restrict__ src)
    {
        //
        // Define the shared memory buffers.
        //
        __shared__ float4 smem_input_buf[SmemInput::size];
        __shared__ float4 smem_output_buf[SmemOutput::size];

        //
        // Define the block tile.
        //
        BlockIdx block_idx(blockIdx.x);
        auto block_n = block_idx.n();
        auto block_p = block_idx.block_p();
        auto block_q = block_idx.block_q();
        auto block_c = block_idx.block_c();

        //
        // Define tile mappings
        //

        // Input to smem.
        Input input(src);
        SmemInput smem_input_store(smem_input_buf);
        bool thread_loads_input;
        int zfill;
        {
            InputIdx idx(threadIdx.x);
            auto block_x = block_q.unfold().cast<X_Dim>() - Block::padding;
            auto x = block_x + idx.x();
            auto c4 = block_c.fold<4>() + idx.c4();

            smem_input_store = smem_input_store[idx.x()][idx.c4()];
            input = input[block_n][block_p.unfold().cast<Y_Dim>()][x][c4];

            bool x_inbounds = (x >= 0 && x < input.X);
            bool c4_inbounds = (c4 < input.C4);
            bool thread_inbounds = (x_inbounds && c4_inbounds);
            thread_loads_input = threadIdx.x < InputIdx::size;
            zfill = thread_inbounds ? 0 : sizeof(Input::data_type);
        }

        // Input-smem to output-smem.
        ConstSmemInput smem_input_load(reinterpret_cast<const float2 *>(smem_input_buf));
        SmemOutput smem_output_store(reinterpret_cast<float2 *>(smem_output_buf));
        {
            SmemInputLoadIdx idx(threadIdx.x);
            smem_input_load = smem_input_load[idx.q().cast<X_Dim>() + Block::padding][idx.c4()][idx.c2()];
            smem_output_store = smem_output_store[idx.q()][idx.c4()][idx.c2()];
        }

        // Smem to output.
        ConstSmemOutput smem_output_load(smem_output_buf);
        Output output(dst);
        bool thread_stores_output;
        {
            ConstSmemOutput::Index idx(threadIdx.x);
            auto q = block_q.unfold() + idx.q();
            auto c4 = block_c.fold<4>() + idx.c4();

            smem_output_load = smem_output_load[idx.q()][idx.c4()];
            output = output[block_n][block_p.unfold()][q][c4];

            thread_stores_output = q.cast<X_Dim>() < input.X && c4 < Block::c4 && threadIdx.x < ConstSmemOutput::Index::size;
        }

        //
        //  Define pipeline stages.
        //
        constexpr unsigned LOAD_INPUT_STAGE = 1 << 0;
        constexpr unsigned COPY_STAGE = 1 << 1;
        constexpr unsigned NUM_STAGES = 2;

        auto num_p = min(block_p.stride, input.Y.cast<P_Dim>() - block_p.unfold());
        int num_iters = num_p.get() + NUM_STAGES - 1;
        int ping_pong = 0;

        Pipeline pipeline;

        //
        // Run the pipeline.
        //      
        for (int iter = 0; iter < num_iters; ++iter)
        {
            pipeline.step(iter < num_p.get());
            if (pipeline.active(LOAD_INPUT_STAGE))
            {
                if (thread_loads_input)
                {
                    __pipeline_memcpy_async(
                        smem_input_store.ping_pong(ping_pong).get(),
                        input.get(),
                        sizeof(Input::data_type),
                        zfill);
                }
                __pipeline_commit();
                input = input.y(1);
            }
            ping_pong = 1 - ping_pong;
            if (pipeline.active(COPY_STAGE))
            {
                __pipeline_wait_prior(pipeline.active(LOAD_INPUT_STAGE) ? 1 : 0);
                __syncthreads();
                *smem_output_store = *smem_input_load.ping_pong(ping_pong);
                __syncthreads();
                if (thread_stores_output)
                {
                    *output = *smem_output_load;
                }
                output = output.p(1);
            }
        }
    }
}
