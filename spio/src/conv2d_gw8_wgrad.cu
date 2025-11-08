#include "spio.cuh"
#include "spio/pipeline.h"

#include "parameters.h"

using namespace spio;
using namespace Params;

extern "C"
{
    __global__ void SPIO_CONV_WGRAD_KERNEL(
        float *__restrict__ wgrad_ptr,
        const uint4 *__restrict__ input_ptr,
        const uint4 *__restrict__ deltas_ptr)
    {
        //
        // Allocate the shared memory tensors.
        //
        static_assert(SmemInput::element_size == SmemDelta::element_size,
                      "SmemInput and SmemDelta assumed to have the same element size.");
        constexpr int smem_size = spio::max(
            SmemInput::storage_size() + SmemDelta::storage_size(),
            SmemWgrad::num_bytes() / SmemInput::element_size);
        __shared__ SmemInput::data_type smem[smem_size];

        StackAllocator smem_alloc(smem);
        auto smem_input = SmemInput::allocate(smem_alloc);
        auto smem_delta = SmemDelta::allocate(smem_alloc);

        //
        // Define the block tile.
        //
        BlockIdx block_idx(blockIdx.x);

        //
        // Define tile mappings.
        //

        // Load input to smem.
        Input::base_cursor_type global_input;
        SmemInput::base_cursor_type smem_input_store;
        bool thread_loads_input;
        bool input_inbounds;
        auto input_n = [&]()
        {
            InputIdx idx(threadIdx.x);
            smem_input_store = smem_input[idx].rebase();
            auto _input_n = idx.get<N>();
            auto block_x = block_idx.get<BLOCK_Q>().unfold().cast<X>() - PADDING_W;
            auto x = block_x + idx.get<X>();
            auto c8 = block_idx.get<BLOCK_C>().fold<8>() + idx.get<C8>();
            global_input = Input(input_ptr)[x][c8].rebase();
            thread_loads_input = threadIdx.x < InputIdx::size();
            bool x_inbounds = (x >= 0 && x < Input::size<X>());
            bool c8_inbounds = (c8 < Input::size<C8>());
            input_inbounds = (x_inbounds && c8_inbounds);
            return _input_n;
        }();

        // Load delta to smem.
        Delta::base_cursor_type global_delta;
        SmemDelta::base_cursor_type smem_delta_store;
        bool delta_inbounds;
        bool thread_loads_delta;
        auto delta_n = [&]()
        {
            DeltaIdx idx(threadIdx.x);
            smem_delta_store = smem_delta[idx].rebase();
            auto q = block_idx.get<BLOCK_Q>().unfold() + idx.get<Q>();
            auto k8 = block_idx.get<BLOCK_C>().fold<8>().cast<K>() + idx.get<K8>();
            global_delta = Delta(deltas_ptr)[q][k8].rebase();
            thread_loads_delta = threadIdx.x < DeltaIdx::size();
            delta_inbounds = (k8 < Delta::size<K8>() && q < Delta::size<Q>());
            return idx.get<N>();
        }();

        // Load input-smem to register.
        SmemInput::base_cursor_type smem_input_load;
        {
            SmemInputLoadIdx idx(threadIdx.x);
            auto x = idx.get<Q>().cast<X>() + idx.get<WARP_S>().unfold().cast<X>() + idx.get<S>().cast<X>();
            smem_input_load = smem_input[x][idx.get<C8>()].rebase();
        }

        // Load delta-smem to register.
        SmemDelta::base_cursor_type smem_delta_load = smem_delta[SmemDeltaLoadIdx(threadIdx.x)].rebase();

        //
        // Declare the accumulators.
        //
        AccTensor::data_type acc_array[AccTensor::storage_size()];
        AccTensor acc(acc_array);
        acc.zero();

        // Iterate over batches.
        for (int n_iter = 0; n_iter < BLOCK_N_ITERS; ++n_iter)
        {
            //
            // Define the pipeline.
            //
            Pipeline pipeline;
            constexpr unsigned STAGE_GLOBAL_DELTAS_LOAD = 1 << 0;
            constexpr unsigned STAGE_SMEM_DELTAS_LOAD = 1 << 1;
            constexpr unsigned STAGE_GLOBAL_INPUT_LOAD = 1 << (R_Size - 1);
            constexpr unsigned STAGE_COMPUTE = 1 << R_Size;
            constexpr int NUM_ITERS = R_Size + BLOCK_Y(1).unfold().get();

            //
            // Define the input and delta (grad_output) pointers for the current batch iteration.
            //
            auto input_n_iter = block_idx.get<BLOCK_N>().unfold() + input_n + n_iter * WARP_N_Size;
            bool input_n_inbounds = (input_n_iter < Input::size<N>());
            auto global_input_n_iter = global_input[input_n_iter].rebase();

            auto delta_n_iter = block_idx.get<BLOCK_N>().unfold() + delta_n + n_iter * WARP_N_Size;
            bool delta_n_inbounds = (delta_n_iter < Delta::size<N>());
            auto global_delta_n_iter = global_delta[delta_n_iter].rebase();

            int y = block_idx.get<BLOCK_Y>().unfold().get();
            int p = y - TRANSPOSE_PADDING_H;
            int ping_pong = 0;

            //
            // Define the deltas fragments.
            //
            DeltaTensor::data_type delta_array[DeltaTensor::storage_size()];
            DeltaTensor deltas(delta_array);

            // Run the pipeline, unrolling it R times.
            for (int iter = 0; iter < NUM_ITERS; iter += R_Size)
            {
                for (int phase = 0; phase < R_Size && iter + phase < NUM_ITERS; ++phase)
                {
                    pipeline.step(iter + phase < BLOCK_P(1).unfold().get());
                    if (pipeline.active(STAGE_GLOBAL_INPUT_LOAD, STAGE_GLOBAL_DELTAS_LOAD))
                    {
                        if (thread_loads_input)
                        {
                            memcpy_async(
                                smem_input_store[PING_PONG(ping_pong)].get(),
                                global_input_n_iter[Y(y)].get(),
                                input_inbounds && input_n_inbounds && (y >= 0 && y < Input::size<Y>().get()));
                        }
                        ++y;
                    }
                    if (pipeline.active(STAGE_GLOBAL_DELTAS_LOAD))
                    {
                        if (thread_loads_delta)
                        {
                            memcpy_async(
                                smem_delta_store[PING_PONG(ping_pong)].get(),
                                global_delta_n_iter[P(p)].get(),
                                delta_inbounds && delta_n_inbounds && (p >= 0 && p < Delta::size<P>().get()));
                        }
                        __pipeline_commit();
                        ++p;
                    }
                    ping_pong ^= 1;
                    if (pipeline.active(STAGE_SMEM_DELTAS_LOAD))
                    {
                        __pipeline_wait_prior(pipeline.active(STAGE_GLOBAL_DELTAS_LOAD) ? 1 : 0);
                        __syncthreads();
                        for (auto warp_n : range(deltas.size<N>()))
                        {
                            auto r_idx = R((R_Size - 1 + phase) % R_Size);
                            deltas[warp_n][r_idx]->load_trans(smem_delta_load[PING_PONG(ping_pong)][warp_n].get());
                        }
                    }
                    if (pipeline.active(STAGE_COMPUTE))
                    {
                        for (auto warp_n : range(deltas.size<N>()))
                        {
                            InputTensor::data_type input_array[InputTensor::storage_size()];
                            InputTensor input(input_array);
                            for (auto s2 : range(input.size<S2>()))
                            {
                                input[s2]->load_trans(smem_input_load[PING_PONG(ping_pong)][warp_n][s2.unfold().cast<X>()].get());
                            }
                            for (auto s2 : range(input.size<S2>()))
                            {
                                for (auto r : range(acc.size<R>()))
                                {
                                    mma_trans(*acc[s2][r], *input[s2], *deltas[warp_n][(r + phase) % R_Size], *acc[s2][r]);
                                }
                            }
                        }
                    }
                    if (pipeline.active(STAGE_SMEM_DELTAS_LOAD))
                    {
                        __syncthreads();
                    }
                }
            }
        }

        smem_delta.deallocate(smem_alloc);
        smem_input.deallocate(smem_alloc);
        auto smem_wgrad = SmemWgrad::allocate(smem_alloc);

        // Store accumulator to wgrad-smem.
        auto global_wgrad = Wgrad(wgrad_ptr)[block_idx.get<BLOCK_C>().unfold().cast<K>()].rebase();
        SmemWgrad::base_cursor_type smem_wgrad_store;
        auto warp_s = [&]()
        {
            SmemWgradStoreIdx idx(threadIdx.x);
            auto _warp_s = idx.get<WARP_S>().unfold();
            Acc::Index acc_idx(idx.get<LANE>().get());
            smem_wgrad_store = smem_wgrad[idx.get<K8>()][acc_idx.get<K2>()][_warp_s][acc_idx.get<C>()].rebase();
            return _warp_s;
        }();

        // Note: Out-of-resources CUDA error when coded as a range-based loop (auto r : acc.R)
#pragma unroll R_Size
        for (int r = 0; r < R_Size; ++r)
        {
            if (r > 0)
            {
                __syncthreads();
            }
            for (auto s : range(WARP_S::stride))
            {
                if (warp_s + s >= S_Size)
                {
                    break;
                }
                auto sd2 = s.fold<2>();
                auto sm2 = s % 2;
                *smem_wgrad_store[s] = acc[sd2][R(r)]->fragment(sm2.get());
            }
            __syncthreads();

            // Add wgrad to the global result.
#pragma unroll 1
            for (int iter = threadIdx.x; iter < WgradStoreIdx::size(); iter += Block::threads)
            {
                // Flip r-dimension.
                WgradStoreIdx idx(iter);
                auto smem_wgrad_load_iter = smem_wgrad[idx].rebase();
                auto wgrad_iter = global_wgrad[idx.get<K8>().unfold()][acc.size<R>() - 1 - r][idx.get<S>()][idx.get<C>()];
                auto k = block_idx.get<BLOCK_C>().unfold().cast<K>() + idx.get<K8>().unfold();
#pragma unroll 4
                for (int k2 = 0; k2 < 4; ++k2)
                {
                    if (k + k2 * 2 < Wgrad::size<K>().get())
                    {
                        float2 wgrad_f2 = *smem_wgrad_load_iter[K2(k2)];
                        atomicAdd(wgrad_iter[K(k2 * 2 + 0)].get(), wgrad_f2.x);
                        atomicAdd(wgrad_iter[K(k2 * 2 + 1)].get(), wgrad_f2.y);
                    }
                }
            }
        }
    }
}