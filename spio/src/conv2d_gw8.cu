#include "spio/memory.cuh"
#include "spio/pipeline.h"
#include "spio/fragment.cuh"
#include "spio/fragment_mma.cuh"
#include "spio/mma.cuh"
#include "spio/ldmatrix.cuh"
#include "spio/mathutil.h"

// Include the generated header file that contains tensor, index, parameter and macro definitions.
#include "parameters.h"

using namespace spio;

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
        __shared__ uint4 smem_buf[spio::max(SmemWeights::size, SmemOutput::size)];
        uint4 *smem_weights_buf = smem_buf;
        uint4 *smem_output_buf = smem_buf;

        //
        // Define the block tile.
        //
        BlockIdx block_idx(blockIdx.x);

        // Fetch the bias
        auto bias_f32 = [&]()
        {
            if constexpr (Mode::has_bias)
            {
                BiasIdx idx(threadIdx.x);
                Acc::Index acc_idx(idx.lane());
                auto k8 = idx.k8() + block_idx.block_c().fold<8>().cast<K_Dim>();
                if (k8 < Bias::K8)
                {
                    return *Bias(bias_ptr)[k8][acc_idx.k2()];
                }
            }
            return make_float2(0, 0);
        }();

        //
        // Define tile mappings
        //

        // Weights-smem to registers.
        auto smem_weights_load = [&]()
        {
            SmemWeightsLoadIdx idx(threadIdx.x);
            return ConstSmemWeights(smem_weights_buf)[idx.k8()][idx.k()];
        }();

        // Input to smem.
        Input input(input_ptr);
        SmemInput smem_input_store(smem_input_buf);
        bool thread_loads_input;
        bool z_inbounds;
        {
            InputIdx idx(threadIdx.x);
            smem_input_store = smem_input_store[idx.n()][idx.x()][idx.c8()];

            auto n = block_idx.block_n().unfold() + idx.n();
            auto x = block_idx.block_q().unfold().cast<X_Dim>() + idx.x() - Padding::w;
            auto c8 = block_idx.block_c().fold<8>() + idx.c8();
            input = input[n][x][c8];

            z_inbounds = ((n < input.N) && (x >= 0 && x < input.X) && (c8 < input.C8));
            thread_loads_input = threadIdx.x < idx.size;
        }

        // Input-smem to register.
        auto smem_input_load = [&]()
        {
            SmemInputLoadIdx idx(threadIdx.x);
            return SmemInput(smem_input_buf)[idx.n()][idx.q().cast<X_Dim>()][idx.c8()];
        }();

        // Register to output-smem.
        SmemOutput smem_output_store_qn0;
        SmemOutput smem_output_store_qn8;
        {
            SmemOutputStoreIdx thread_idx(threadIdx.x);
            Acc::Index acc_idx(thread_idx.lane());
            BlockQNIdx block_qn_0_idx(acc_idx.qn(0).get());
            BlockQNIdx block_qn_8_idx(acc_idx.qn(1).get());
            auto smem_output_store = SmemOutput(reinterpret_cast<__half2 *>(smem_output_buf))[thread_idx.k8()][acc_idx.k2()];
            smem_output_store_qn0 = smem_output_store[block_qn_0_idx.n()][block_qn_0_idx.q()];
            smem_output_store_qn8 = smem_output_store[block_qn_8_idx.n()][block_qn_8_idx.q()];
        }

        // Output-smem to output.
        ConstSmemOutput smem_output_load(smem_output_buf);
        Output output(dst);
        bool thread_stores_output;
        {
            OutputStoreIdx idx(threadIdx.x);
            auto q = block_idx.block_q().unfold() + idx.q();
            auto n = block_idx.block_n().unfold() + idx.n();
            auto k8 = block_idx.block_c().fold<8>().cast<K_Dim>() + idx.k8();
            smem_output_load = smem_output_load[idx.n()][idx.q()][idx.k8()];
            output = output[n][block_idx.block_p().unfold()][q][k8];
            thread_stores_output = ((n < output.N) && (q < output.Q) && (k8 < output.K8) && (threadIdx.x < OutputStoreIdx::size));
        }

        // Copy weights from global memory to smem asynchronously.
        auto weight = Weights(weights_ptr).k(block_idx.block_c().unfold().cast<K_Dim>());
        for (int idx = threadIdx.x; idx < SmemWeights::size; idx += Block::threads)
        {
            Weights::Index weight_idx(idx);
            auto k = block_idx.block_c().unfold().cast<K_Dim>() + weight_idx.k();
            memcpy_async(smem_weights_buf + idx, weight.get() + idx, k < weight.K);
        }
        __pipeline_commit();

        // Define the pipeline.
        constexpr unsigned LOAD_INPUT_STAGE = 1 << 0;
        constexpr unsigned COMPUTE_STAGE = 1 << 1;
        constexpr unsigned NUM_STAGES = 2;
        auto num_p = spio::min(BLOCK_P_Dim::stride, Output::P - block_idx.block_p().unfold());
        int num_y = num_p.get() + Weights::R.get() - 1;
        int num_iters = num_y + NUM_STAGES - 1;
        int ping_pong = 0;
        Pipeline pipeline;

        // Prefetch the first input row.
        int y = block_idx.block_p().unfold().get() - Padding::h;
        pipeline.step(0 < num_y);
        if (pipeline.active(LOAD_INPUT_STAGE))
        {
            if (thread_loads_input)
            {
                memcpy_async(
                    smem_input_store.ping_pong(ping_pong).get(),
                    input.y(y).get(),
                    (y >= 0 && y < input.Y.get()) && z_inbounds);
            }
            __pipeline_commit();
            ++y;
        }
        ping_pong ^= 1;

        __pipeline_wait_prior(1);
        __syncthreads();

        // Load weights to registers.
        WeightsReg::data_type wgts_data[WeightsReg::size];
        WeightsReg wgts(wgts_data);
        for (auto r : Weights::R)
        {
            for (auto s : Weights::S)
            {
                // The input-gradient operation uses the transpose of the weights.
                if constexpr (Mode::igrad)
                {
                    wgts[r][s]->load_trans(smem_weights_load[wgts.R - 1 - r][wgts.S - 1 - s].get());
                }
                else
                {
                    wgts[r][s]->load(smem_weights_load[r][s].get());
                }
            }
        }

        AccReg::data_type acc_data[AccReg::size];
        AccReg acc(acc_data);
        for (auto p : acc.P)
        {
            acc[p]->fill(bias_f32);
        }

        // Run the first Weights::R pipeline steps.
        int iter = 1;
        for (auto phase : acc.P)
        {
            pipeline.step(iter + phase.get() < num_y);
            if (pipeline.active(LOAD_INPUT_STAGE))
            {
                if (thread_loads_input)
                {
                    memcpy_async(
                        smem_input_store.ping_pong(ping_pong).get(),
                        input.y(y).get(),
                        (y >= 0 && y < input.Y.get()) && z_inbounds);
                }
                __pipeline_commit();
                ++y;
            }
            ping_pong ^= 1;
            if (pipeline.active(COMPUTE_STAGE))
            {
                __pipeline_wait_prior(pipeline.active(LOAD_INPUT_STAGE) ? 1 : 0);
                __syncthreads();

                auto smem_input_load_iter = smem_input_load.ping_pong(ping_pong);
                for (auto s : wgts.S)
                {
                    auto in = In::load_new(smem_input_load_iter[s.cast<X_Dim>()].get());
                    // Skip r > phase because these contribute to out-of-bounds outputs p < 0.
                    for (auto r : phase.cast<R_Dim>() + 1)
                    {
                        auto p = (acc.P - 1 - r.cast<P_Dim>() + phase) % acc.P;
                        mma_trans(*acc[p], in, *wgts[r][s], *acc[p]);
                    }
                }
                __syncthreads();
            }
        }

        // Store the first output row to shared memory.
        *smem_output_store_qn0 = acc[acc.P - 1]->to_half2(0);
        *smem_output_store_qn8 = acc[acc.P - 1]->to_half2(1);
        acc[acc.P - 1]->fill(bias_f32);
        __syncthreads();

        // Store the first output row to global memory.
        if (num_p > 0)
        {
            if (thread_stores_output)
            {
                *output = *smem_output_load;
            }
            output = output.p(1);
        }

        iter += acc.P.get();

        // Run the main loop over the remaining input rows.
        for (; iter < num_iters; iter += Weights::R.get())
        {
            // Unroll the main loop by acc.P steps.
            for (auto phase : acc.P)
            {
                pipeline.step(iter + phase.get() < num_y);
                if (pipeline.active(LOAD_INPUT_STAGE))
                {
                    if (thread_loads_input)
                    {
                        memcpy_async(
                            smem_input_store.ping_pong(ping_pong).get(),
                            input.y(y).get(),
                            (y >= 0 && y < input.Y.get()) && z_inbounds);
                    }
                    __pipeline_commit();
                    ++y;
                }
                ping_pong ^= 1;
                if (pipeline.active(COMPUTE_STAGE))
                {
                    __pipeline_wait_prior(pipeline.active(LOAD_INPUT_STAGE) ? 1 : 0);
                    __syncthreads();
                    auto smem_input_load_iter = smem_input_load.ping_pong(ping_pong);
                    for (auto s : wgts.S)
                    {
                        auto in = In::load_new(smem_input_load_iter[s.cast<X_Dim>()].get());
                        for (auto r : wgts.R)
                        {
                            auto p = (acc.P - 1 - r.cast<P_Dim>() + phase) % acc.P;
                            mma_trans(*acc[p], in, *wgts[r][s], *acc[p]);
                        }
                    }
                    *smem_output_store_qn0 = acc[phase]->to_half2(0);
                    *smem_output_store_qn8 = acc[phase]->to_half2(1);
                    acc[phase]->fill(bias_f32);
                    __syncthreads();

                    auto store_p = P_Dim(iter) + phase - acc.P;
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
