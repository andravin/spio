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
        __shared__ float4 smem_input_buf[SmemInput::size()];
        __shared__ float4 smem_output_buf[SmemOutput::size()];

        //
        // Define the block tile.
        //
        BlockIdx block_idx(blockIdx.x);
        auto block_n = block_idx.get<N>();
        auto block_p = block_idx.get<BLOCK_P>();
        auto block_q = block_idx.get<BLOCK_Q>();
        auto block_c = block_idx.get<BLOCK_C>();

        //
        // Define tile mappings
        //

        // Input to smem.
        Input::cursor_type input;
        SmemInput::cursor_type smem_input_store;
        bool thread_loads_input;
        int zfill;
        {
            InputIdx idx(threadIdx.x);
            auto block_x = block_q.unfold().cast<X>() - Block::padding;
            auto x = block_x + idx.get<X>();
            auto c4 = block_c.fold<4>() + idx.get<C4>();

            smem_input_store = SmemInput(smem_input_buf)[idx.get<X>()][idx.get<C4>()];
            input = Input(src)[block_n][block_p.unfold().cast<Y>()][x][c4];

            bool x_inbounds = (x >= 0 && x < Input::size<X>());
            bool c4_inbounds = (c4 < Input::size<C4>());
            bool thread_inbounds = (x_inbounds && c4_inbounds);
            thread_loads_input = threadIdx.x < InputIdx::total_size;
            zfill = thread_inbounds ? 0 : sizeof(Input::data_type);
        }

        // Input-smem to output-smem.
        ConstSmemInput::cursor_type smem_input_load;
        SmemOutput::cursor_type smem_output_store;
        {
            SmemInputLoadIdx idx(threadIdx.x);
            auto q_dim = idx.get<Q>();
            auto c4_dim = idx.get<C4>();
            auto c2_dim = idx.get<C2>();
            
            smem_input_load = ConstSmemInput(reinterpret_cast<const float2 *>(smem_input_buf))[q_dim.cast<X>() + Block::padding][c4_dim][c2_dim];
            smem_output_store = SmemOutput(reinterpret_cast<float2 *>(smem_output_buf))[q_dim][c4_dim][c2_dim];
        }

        // Smem to output.
        ConstSmemOutput::cursor_type smem_output_load;
        Output::cursor_type output;
        bool thread_stores_output;
        {
            auto idx = ConstSmemOutput::index_type(threadIdx.x);
            auto q = block_q.unfold() + idx.get<Q>();
            auto c4 = block_c.fold<4>() + idx.get<C4>();

            smem_output_load = ConstSmemOutput(smem_output_buf)[idx.get<Q>()][idx.get<C4>()];
            output = Output(dst)[block_n][block_p.unfold()][q][c4];

            thread_stores_output = q.cast<X>() < Input::size<X>() && 
                                  c4 < Block::c4 && 
                                  threadIdx.x < ConstSmemOutput::size();
        }

        //
        //  Define pipeline stages.
        //
        constexpr unsigned LOAD_INPUT_STAGE = 1 << 0;
        constexpr unsigned COPY_STAGE = 1 << 1;
        constexpr unsigned NUM_STAGES = 2;

        auto num_p = min(block_p.stride, Input::size<Y>().cast<P>() - block_p.unfold());
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
                        smem_input_store[PING_PONG(ping_pong)].get(),
                        input.get(),
                        sizeof(Input::data_type),
                        zfill);
                }
                __pipeline_commit();
                input = input[Y(1)];
            }
            ping_pong = 1 - ping_pong;
            if (pipeline.active(COPY_STAGE))
            {
                __pipeline_wait_prior(pipeline.active(LOAD_INPUT_STAGE) ? 1 : 0);
                __syncthreads();
                *smem_output_store = *smem_input_load[PING_PONG(ping_pong)];
                __syncthreads();
                if (thread_stores_output)
                {
                    *output = *smem_output_load;
                }
                output = output[P(1)];
            }
        }
    }
}
