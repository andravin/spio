#include <cuda_pipeline.h>

#include "spio/pipeline.h"
#include "spio/mma.cuh"
#include "spio/ldmatrix.cuh"

// Include the generated header file that contains tensor, index, parameter and macro definitions.
#include "parameters.h"

using namespace spio;

using WeightsReg = _MMA_M16_N8_F16_B<Weights::R * Weights::S>;

namespace
{
    /// @brief  Return the maximum of two integers. Is a constexpr.
    __device__ constexpr int _max(int a, int b)
    {
        return a > b ? a : b;
    }

    /// @brief igrad use the transpose of the weights.
    __device__ inline uint4 ld_weights_x4(const void *p)
    {
        if constexpr (Mode::igrad)
        {
            return ldmatrix_x4_trans(p);
        }
        else
        {
            return ldmatrix_x4(p);
        }
    }

    /// @brief igrad use the transpose of the weights.
    __device__ inline unsigned ld_weights_x1(const void *p)
    {
        if constexpr (Mode::igrad)
        {
            return ldmatrix_x1_trans(p);
        }
        else
        {
            return ldmatrix_x1(p);
        }
    }

    /// @brief igrad uses the transpose of the weights.
    __device__ inline unsigned get_weight(const WeightsReg &wgts, int r, int s)
    {
        if constexpr (Mode::igrad)
        {
            return wgts.reg((Weights::R - 1 - r) * Weights::S + (Weights::S - 1 - s));
        }
        else
        {
            return wgts.reg(r * Weights::S + s);
        }
    }
}

extern "C"
{
    __global__ void SPIO_CONV_KERNEL(
        uint4 *__restrict__ dst,
        const uint4 *__restrict__ input_ptr,
        const uint4 *__restrict__ weights_ptr,
        const float2 *__restrict__ bias_ptr)
    {
        //
        // Define the shared memory buffers.
        //
        __shared__ uint4 smem_input_buf[SmemInput::size];
        __shared__ uint4 smem_buf[_max(SmemWeights::size, SmemOutput::size)];
        uint4 *smem_weights_buf = smem_buf;
        uint4 *smem_output_buf = smem_buf;

        //
        // Define the block tile.
        //
        BlockIdx block_idx(blockIdx.x);
        int block_n = block_idx.n() * Block::n;
        int block_p = block_idx.p() * Block::p;
        int block_q = block_idx.q() * Block::q;
        int block_c8 = block_idx.c8() * Block::c8;

        int block_c = block_c8 * 8;

        // Fetch the bias
        float2 bias_f32 = make_float2(0, 0);
        if constexpr (Mode::has_bias)
        {
            BiasIdx idx(threadIdx.x);
            int lane_k2 = Acc::k2(idx.lane());
            int k8 = block_c8 + idx.k8();
            if (k8 < Output::K8) {
                bias_f32 = *Bias(bias_ptr).k8(k8).k2(lane_k2);
            }
        }

        //
        // Define tile mappings
        //

        // Weights-smem to registers.
        ConstSmemWeights smem_weights_load(smem_weights_buf);
        {
            SmemWeightsLoadIdx idx(threadIdx.x);
            smem_weights_load = smem_weights_load.kd8(idx.kd8()).km8(idx.km8()).rs(idx.rs());
        }

        // Input to smem.
        Input input(input_ptr);
        SmemInput smem_input_store(smem_input_buf);
        bool thread_loads_input;
        int zfill;
        {
            InputIdx idx(threadIdx.x);
            int block_x = block_q - Block::padding_w;
            int n = block_n + idx.n();
            int x = block_x + idx.x();
            int c8 = block_c8 + idx.c8();

            smem_input_store = smem_input_store.n(idx.n()).x(idx.x()).c8(idx.c8());
            input = input.n(n).x(x).c8(c8);

            bool n_inbounds = (n < Input::N);
            bool x_inbounds = (x >= 0 && x < Input::X);
            bool c8_inbounds = (c8 < Input::C8);
            bool thread_inbounds = (n_inbounds && x_inbounds && c8_inbounds);
            thread_loads_input = threadIdx.x < InputIdx::size;
            zfill = thread_inbounds ? 0 : sizeof(Input::data_type);
        }

        // Input-smem to register.
        SmemInput smem_input_load(smem_input_buf);
        {
            SmemInputLoadIdx idx(threadIdx.x);
            smem_input_load = smem_input_load.n(idx.n()).x(idx.q()).c8(idx.c8());
        }

        // Register to output-smem.
        SmemOutput smem_output_store_qn0;
        SmemOutput smem_output_store_qn8;
        {
            SmemOutputStoreIdx idx(threadIdx.x);
            int lane_qn_0 = Acc::qn(idx.lane(), 0);
            OutputQNIdx lane_qn_0_idx(lane_qn_0);
            int lane_q_0 = lane_qn_0_idx.q();
            int lane_n_0 = lane_qn_0_idx.n();

            int lane_qn_8 = Acc::qn(idx.lane(), 1);
            OutputQNIdx lane_qn_8_idx(lane_qn_8);
            int lane_q_8 = lane_qn_8_idx.q();
            int lane_n_8 = lane_qn_8_idx.n();

            int lane_k2 = Acc::k2(idx.lane());
            
            auto smem_output_store = SmemOutput(reinterpret_cast<__half2 *>(smem_output_buf)).k8(idx.k8()).k2(lane_k2);
            smem_output_store_qn0 = smem_output_store.n(lane_n_0).q(lane_q_0);
            smem_output_store_qn8 = smem_output_store.n(lane_n_8).q(lane_q_8);
        }

        // Output-smem to output.
        ConstSmemOutput smem_output_load(smem_output_buf);
        Output output(dst);
        bool thread_stores_output;
        {
            OutputStoreIdx idx(threadIdx.x);
            int q = block_q + idx.q();
            int n = block_n + idx.n();
            int k8 = block_c8 + idx.k8();
            smem_output_load = smem_output_load.n(idx.n()).q(idx.q()).k8(idx.k8());
            output = output.n(n).p(block_p).q(q).k8(k8);
            thread_stores_output = n < Output::N && q < Output::Q && k8 < Output::K8 && threadIdx.x < OutputStoreIdx::size;
        }

        //
        // Copy weights to smem asynchronously.
        //
        auto weight = Weights(weights_ptr).k(block_c);
        for (int idx = threadIdx.x; idx < SmemWeights::size; idx += Block::threads)
        {
            Weights::Index weight_idx(idx);
            int k = block_c + weight_idx.k();
            int zfill = k < Weights::K ? 0 : sizeof(Weights::data_type);            
            __pipeline_memcpy_async(
                smem_weights_buf + idx,
                weight.get() + idx,
                16,
                zfill);
        }
        __pipeline_commit();

        //
        // Define the pipeline.
        //
        constexpr unsigned LOAD_INPUT_STAGE = 1 << 0;
        constexpr unsigned COMPUTE_STAGE = 1 << 1;
        constexpr unsigned NUM_STAGES = 2;

        int num_p = min(Block::p, Output::P - block_p);
        int num_y = num_p + Weights::R - 1;
        int num_iters = num_y + NUM_STAGES - 1;
        int ping_pong = 0;

        int y = block_p - Block::padding_h;
        input = input.y(y);

        Pipeline pipeline;

        //
        // Run the first pipeline step.
        //
        pipeline.step(0 < num_y);
        if (pipeline.active(LOAD_INPUT_STAGE))
        {
            bool y_inbounds = (y >= 0 && y < Input::Y);
            int y_fill = (y_inbounds) ? 0 : sizeof(Input::data_type);
            if (thread_loads_input)
            {
                __pipeline_memcpy_async(
                    smem_input_store.ping_pong(ping_pong).get(),
                    input.get(),
                    sizeof(Input::data_type),
                    zfill | y_fill);
            }
            __pipeline_commit();
            ++y;
            input = input.y(1);
        }
        ping_pong = 1 - ping_pong;

        //
        // Copy weights from smem to registers.
        //
        //
        WeightsReg wgts;
        __pipeline_wait_prior(1);
        __syncthreads();
        int rs = 0;
        for (; rs < Weights::R * Weights::S; rs += 4)
        {
            wgts.reg4(rs / 4) = ld_weights_x4(smem_weights_load.rs(rs).get());
        }
        for (; rs < Weights::R * Weights::S; ++rs)
        {
            wgts.reg(rs) = ld_weights_x1(smem_weights_load.rs(rs).get());
        }


        //
        // Declare the accumulators.
        //
        Acc acc[Weights::R];
        for (int p = 0; p < Weights::R; ++p)
        {
            acc[p].fill(bias_f32);
        }

        //
        // Run the first Weights::R pipeline steps.
        //
        // Skip filter-rows that contribute to out-of-bounds outputs.
        //
        int iter = 1;
        for (int phase = 0; phase < Weights::R; ++phase)
        {
            pipeline.step(iter + phase < num_y);
            if (pipeline.active(LOAD_INPUT_STAGE))
            {
                bool y_inbounds = (y >= 0 && y < Input::Y);
                int y_fill = (y_inbounds) ? 0 : sizeof(Input::data_type);
                if (thread_loads_input)
                {
                    __pipeline_memcpy_async(
                        smem_input_store.ping_pong(ping_pong).get(),
                        input.get(),
                        sizeof(Input::data_type),
                        zfill | y_fill);
                }
                __pipeline_commit();
                ++y;
                input = input.y(1);
            }
            ping_pong = 1 - ping_pong;
            if (pipeline.active(COMPUTE_STAGE))
            {
                __pipeline_wait_prior(pipeline.active(LOAD_INPUT_STAGE) ? 1 : 0);
                __syncthreads();

                auto smem_input_load_iter = smem_input_load.ping_pong(ping_pong);
                for (int s = 0; s < Weights::S; ++s)
                {
                    // Load shift of input row.
                    MMA_M16_K16_F16_A in;
                    in.reg2() = ldmatrix_x2(smem_input_load_iter.x(s).get());

                    // Multiply shifted row by column of weights, updating the corresponding output row.
                    //
                    // Skip r > phase because these contribute to out-of-bounds outputs p < 0.
                    for (int r = 0; r <= phase; ++r)
                    {
                        int p = Weights::R - 1 - r + phase;
                        mma_m16_n8_k8(acc[p % Weights::R].vec4(), in.reg2(), get_weight(wgts, r, s), acc[p % Weights::R].vec4());
                    }
                }
                __syncthreads();
            }
        }

        // Store the first output row to shared memory.
        *smem_output_store_qn0 = acc[Weights::R - 1].to_half2(0);
        *smem_output_store_qn8 = acc[Weights::R - 1].to_half2(1);
        acc[Weights::R - 1].fill(bias_f32);

        // If the first output row is inbounds store it to global memory
        __syncthreads();
        if (0 < num_p)
        {
            if (thread_stores_output)
            {
                *output = *smem_output_load;
            }
            output = output.p(1);
        }

        iter += Weights::R;

        for (; iter < num_iters; iter += Weights::R)
        {
            for (int phase = 0; phase < Weights::R; ++phase)
            {
                pipeline.step(iter + phase < num_y);
                if (pipeline.active(LOAD_INPUT_STAGE))
                {
                    bool y_inbounds = (y >= 0 && y < Input::Y);
                    int y_fill = (y_inbounds) ? 0 : sizeof(Input::data_type);
                    if (thread_loads_input)
                    {
                        __pipeline_memcpy_async(
                            smem_input_store.ping_pong(ping_pong).get(),
                            input.get(),
                            sizeof(Input::data_type),
                            zfill | y_fill);
                    }
                    __pipeline_commit();
                    ++y;
                    input = input.y(1);
                }
                ping_pong = 1 - ping_pong;
                if (pipeline.active(COMPUTE_STAGE))
                {
                    __pipeline_wait_prior(pipeline.active(LOAD_INPUT_STAGE) ? 1 : 0);
                    __syncthreads();
                    auto smem_input_load_iter = smem_input_load.ping_pong(ping_pong);
                    for (int s = 0; s < Weights::S; ++s)
                    {
                        // Load shift of input row.
                        MMA_M16_K16_F16_A in;
                        in.reg2() = ldmatrix_x2(smem_input_load_iter.x(s).get());

                        // Multiply the "s"-shifted row by column "s" of the kernel, updating the corresponding output row.
                        for (int r = 0; r < Weights::R; ++r)
                        {
                            int p = Weights::R - 1 - r + phase;
                            mma_m16_n8_k8(acc[p % Weights::R].vec4(), in.reg2(), get_weight(wgts, r, s), acc[p % Weights::R].vec4());
                        }
                    }
                    *smem_output_store_qn0 = acc[phase].to_half2(0);
                    *smem_output_store_qn8 = acc[phase].to_half2(1);
                    acc[phase].fill(bias_f32);
                    __syncthreads();

                    // If the current output row is inbounds store it.
                    int store_p = iter + phase - Weights::R;
                    if (store_p < num_p && thread_stores_output)
                    {
                        *output = *smem_output_load;
                    }
                    output = output.p(1);
                }
            }
        }
    }
}
