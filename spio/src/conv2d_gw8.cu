#include "spio.cuh"
#include "spio/pipeline.h"

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
        // Create the shared memory allocator.
        //
        constexpr int smem_size = SmemInput::storage_size() + spio::max(SmemWeights::storage_size(), SmemOutput::storage_size());
        __shared__ uint4 smem[smem_size];
        StackAllocator smem_alloc(smem);

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
                Acc::Index acc_idx(idx.get<LANE>().get());
                auto k8 = idx.get<K8>() + block_idx.get<BLOCK_C>().fold<8>().cast<K>();
                if (k8 < Bias::size<K8>())
                {
                    return *Bias(bias_ptr)[k8][acc_idx.get<K2>()];
                }
            }
            return make_float2(0, 0);
        }();

        //
        // Define tile mappings
        //

        // Map input to smem.
        auto smem_input = SmemInput::allocate(smem_alloc);
        Input::base_cursor_type input;
        SmemInput::base_cursor_type smem_input_store;
        bool thread_loads_input;
        bool z_inbounds;
        {
            InputIdx idx(threadIdx.x);
            smem_input_store = smem_input[idx].rebase();
            auto n = block_idx.get<BLOCK_N>().unfold() + idx.get<N>();
            auto x = block_idx.get<BLOCK_Q>().unfold().cast<X>() + idx.get<X>() - Padding::w;
            auto c8 = block_idx.get<BLOCK_C>().fold<8>() + idx.get<C8>();
            input = Input(input_ptr)[n][x][c8].rebase();
            z_inbounds = ((n < Input::size<N>()) && (x >= 0 && x < Input::size<X>()) && (c8 < Input::size<C8>()));
            thread_loads_input = threadIdx.x < idx.size();
        }

        // Map weights-smem to registers.
        auto smem_weights = ConstSmemWeights::allocate(smem_alloc);
        auto smem_weights_load = smem_weights[SmemWeightsLoadIdx(threadIdx.x)];

        // Map input-smem to register.
        auto smem_input_load = smem_input[SmemInputLoadIdx(threadIdx.x)].rebase();

        // Copy weights from global memory to smem asynchronously.
        auto weight = Weights(weights_ptr)[block_idx.get<BLOCK_C>().unfold().cast<K>()];
        for (int idx = threadIdx.x; idx < SmemWeights::size(); idx += Block::threads)
        {
            Weights::index_type weight_idx(idx);
            auto k = block_idx.get<BLOCK_C>().unfold().cast<K>() + weight_idx.get<K>();
            memcpy_async(const_cast<uint4 *>(smem_weights.get()) + idx, weight.get() + idx, k < Weights::size<K>());
        }
        __pipeline_commit();

        // Define the pipeline.
        constexpr unsigned LOAD_INPUT_STAGE = 1 << 0;
        constexpr unsigned COMPUTE_STAGE = 1 << 1;
        constexpr unsigned NUM_STAGES = 2;
        auto num_p = spio::min(BLOCK_P::stride, Output::size<P>() - block_idx.get<BLOCK_P>().unfold());
        int num_y = num_p.get() + Weights::size<R>().get() - 1;
        int num_iters = num_y + NUM_STAGES - 1;
        bool ping_pong = false;
        Pipeline pipeline;

        // Prefetch the first input row.
        int y = block_idx.get<BLOCK_P>().unfold().get() - Padding::h;
        pipeline.step(0 < num_y);
        if (pipeline.active(LOAD_INPUT_STAGE))
        {
            if (thread_loads_input)
            {
                memcpy_async(
                    smem_input_store[PING_PONG(ping_pong)].get(),
                    input[Y(y)].get(),
                    (y >= 0 && y < Input::size<Y>().get()) && z_inbounds);
            }
            __pipeline_commit();
            ++y;
        }
        ping_pong = !ping_pong;

        __pipeline_wait_prior(1);
        __syncthreads();

        // Load weights to registers.
        WeightsReg::data_type wgts_data[WeightsReg::storage_size()];
        WeightsReg wgts(wgts_data);
        for (auto r : range(Weights::size<R>()))
        {
            for (auto s : range(Weights::size<S>()))
            {
                // The input-gradient operation uses the transpose of the weights.
                if constexpr (Mode::igrad)
                {
                    wgts[r][s]->load_trans(smem_weights_load[Weights::size<R>() - 1 - r][Weights::size<S>() - 1 - s].get());
                }
                else
                {
                    wgts[r][s]->load(smem_weights_load[r][s].get());
                }
            }
        }
        smem_weights.deallocate(smem_alloc);

        // Map registers to output-smem.
        auto smem_output = SmemOutput::allocate(smem_alloc);
        SmemOutput::base_cursor_type smem_output_store_qn0;
        SmemOutput::base_cursor_type smem_output_store_qn8;
        {
            SmemOutputStoreIdx thread_idx(threadIdx.x);
            Acc::Index acc_idx(thread_idx.get<LANE>().get());
            BlockQNIdx block_qn_0_idx(acc_idx.get<QN>(0).get());
            BlockQNIdx block_qn_8_idx(acc_idx.get<QN>(1).get());
            auto smem_output_store = smem_output[thread_idx.get<K8>()][acc_idx.get<K2>()];
            smem_output_store_qn0 = smem_output_store[block_qn_0_idx].rebase();
            smem_output_store_qn8 = smem_output_store[block_qn_8_idx].rebase();
        }

        // Map output-smem to global output.
        ConstSmemOutput::base_cursor_type smem_output_load;
        Output::base_cursor_type output;
        bool thread_stores_output;
        {
            OutputStoreIdx idx(threadIdx.x);
            smem_output_load = ConstSmemOutput(reinterpret_cast<ConstSmemOutput::data_type *>(smem_output.get()))[idx].rebase();
            auto q = block_idx.get<BLOCK_Q>().unfold() + idx.get<Q>();
            auto n = block_idx.get<BLOCK_N>().unfold() + idx.get<N>();
            auto k8 = block_idx.get<BLOCK_C>().fold<8>().cast<K>() + idx.get<K8>();
            output = Output(dst)[n][block_idx.get<BLOCK_P>().unfold()][q][k8].rebase();
            thread_stores_output = ((n < Output::size<N>()) && (q < Output::size<Q>()) && (k8 < Output::size<K8>()) && (threadIdx.x < OutputStoreIdx::size()));
        }

        // Initialize the accumulators with the bias vector.
        AccReg::data_type acc_data[AccReg::storage_size()];
        AccReg acc(acc_data);
        acc.fill(bias_f32);

        // Run the first Weights::R pipeline steps.
        int iter = 1;
        for (auto phase : range(acc.size<P>()))
        {
            pipeline.step(iter + phase.get() < num_y);
            if (pipeline.active(LOAD_INPUT_STAGE))
            {
                if (thread_loads_input)
                {
                    memcpy_async(
                        smem_input_store[PING_PONG(ping_pong ? 1 : 0)].get(),
                        input[Y(y)].get(),
                        (y >= 0 && y < Input::size<Y>().get()) && z_inbounds);
                }
                __pipeline_commit();
                ++y;
            }
            ping_pong = !ping_pong;
            if (pipeline.active(COMPUTE_STAGE))
            {
                __pipeline_wait_prior(pipeline.active(LOAD_INPUT_STAGE) ? 1 : 0);
                __syncthreads();

                auto smem_input_load_iter = smem_input_load[PING_PONG(ping_pong ? 1 : 0)].rebase();
                for (auto s : range(Weights::size<S>()))
                {
                    auto in = In::load_new(smem_input_load_iter[s.cast<X>()].get());
                    // Skip r > phase because these contribute to out-of-bounds outputs p < 0.
                    for (auto r : range(phase.cast<R>() + 1))
                    {
                        auto p = (acc.size<P>() - 1 - r.cast<P>() + phase) % acc.size<P>();
                        mma_trans(*acc[p], in, *wgts[r][s], *acc[p]);
                    }
                }
                __syncthreads();
            }
        }

        // Store the first output row to shared memory.
        *smem_output_store_qn0 = acc[acc.size<P>() - 1]->to_half2(0);
        *smem_output_store_qn8 = acc[acc.size<P>() - 1]->to_half2(1);
        acc[acc.size<P>() - 1]->fill(bias_f32);
        __syncthreads();

        // Store the first output row to global memory.
        if (num_p > 0)
        {
            if (thread_stores_output)
            {
                *output = *smem_output_load;
            }
            output.step<P>();
        }

        iter += acc.size<P>().get();

        // Run the main loop over the remaining input rows.
        for (; iter < num_iters; iter += Weights::size<R>().get())
        {
            // Unroll the main loop by acc.P steps.
            for (auto phase : range(acc.size<P>()))
            {
                pipeline.step(iter + phase.get() < num_y);
                if (pipeline.active(LOAD_INPUT_STAGE))
                {
                    if (thread_loads_input)
                    {
                        memcpy_async(
                            smem_input_store[PING_PONG(ping_pong ? 1 : 0)].get(),
                            input[Y(y)].get(),
                            (y >= 0 && y < Input::size<Y>().get()) && z_inbounds);
                    }
                    __pipeline_commit();
                    ++y;
                }
                ping_pong = !ping_pong;
                if (pipeline.active(COMPUTE_STAGE))
                {
                    __pipeline_wait_prior(pipeline.active(LOAD_INPUT_STAGE) ? 1 : 0);
                    __syncthreads();
                    auto smem_input_load_iter = smem_input_load[PING_PONG(ping_pong ? 1 : 0)].rebase();
                    for (auto s : range(Weights::size<S>()))
                    {
                        auto in = In::load_new(smem_input_load_iter[s.cast<X>()].get());
                        for (auto r : range(wgts.size<R>()))
                        {
                            auto p = (acc.size<P>() - 1 - r.cast<P>() + phase) % acc.size<P>();
                            mma_trans(*acc[p], in, *wgts[r][s], *acc[p]);
                        }
                    }
                    *smem_output_store_qn0 = acc[phase]->to_half2(0);
                    *smem_output_store_qn8 = acc[phase]->to_half2(1);
                    acc[phase]->fill(bias_f32);
                    __syncthreads();

                    auto store_p = P(iter) + phase - acc.size<P>();
                    if (store_p < num_p && thread_stores_output)
                    {
                        *output = *smem_output_load;
                    }
                    output.step<P>();
                }
            }
        }
    }
}
