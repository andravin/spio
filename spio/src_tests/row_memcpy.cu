#include <cuda_pipeline.h>

#include "spio/pipeline.h"

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
        int block_n = block_idx.n();
        int block_p = block_idx.p() * Block::p;
        int block_q = block_idx.q() * Block::q;
        int block_c4 = block_idx.c4() * Block::c4;

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
            int block_x = block_q - Block::padding;
            int x = block_x + idx.x();
            int c4 = block_c4 + idx.c4();

            smem_input_store = smem_input_store.x(idx.x()).c4(idx.c4());
            input = input.n(block_n).y(block_p).x(x).c4(c4);

            bool x_inbounds = (x >= 0 && x < Input::X);
            bool c4_inbounds = (c4 < Input::C4);
            bool thread_inbounds = (x_inbounds && c4_inbounds);
            thread_loads_input = threadIdx.x < InputIdx::size;
            zfill = thread_inbounds ? 0 : sizeof(Input::data_type);
        }

        // Input-smem to output-smem.
        ConstSmemInput smem_input_load(reinterpret_cast<const float2 *>(smem_input_buf));
        SmemOutput smem_output_store(reinterpret_cast<float2 *>(smem_output_buf));
        {
            SmemInputLoadIdx idx(threadIdx.x);
            smem_input_load = smem_input_load.x(idx.q() + Block::padding).c4(idx.c4()).c2(idx.c2());
            smem_output_store = smem_output_store.q(idx.q()).c4(idx.c4()).c2(idx.c2());
        }

        // Smem to output.
        ConstSmemOutput smem_output_load(smem_output_buf);
        Output output(dst);
        bool thread_stores_output;
        {
            ConstSmemOutput::Index idx(threadIdx.x);
            int q = block_q + idx.q();
            int c4 = block_c4 + idx.c4();

            smem_output_load = smem_output_load.q(idx.q()).c4(idx.c4());
            output = output.n(block_n).p(block_p).q(q).c4(c4);

            thread_stores_output = q < Input::X && c4 < Block::c4 && threadIdx.x < ConstSmemOutput::Index::size;
        }

        //
        //  Define pipeline stages.
        //
        constexpr unsigned LOAD_INPUT_STAGE = 1 << 0;
        constexpr unsigned COPY_STAGE = 1 << 1;
        constexpr unsigned NUM_STAGES = 2;

        int num_p = min(Block::p, Input::Y - block_p);
        int num_iters = num_p + NUM_STAGES - 1;
        int ping_pong = 0;

        Pipeline pipeline;

        //
        // Run the pipeline.
        //      
        for (int iter = 0; iter < num_iters; ++iter)
        {
            pipeline.step(iter < num_p);
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
