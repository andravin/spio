#include <cuda_fp16.h>

#include "spio/memory.cuh"
#include "spio/pipeline.h"
#include "spio/fragment.cuh"
#include "spio/mma.cuh"
#include "spio/ldmatrix.cuh"
#include "spio/mathutil.h"
#include "spio/fragment_mma.cuh"

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
        // Define the shared memory buffers.
        //
        // Overlap the wgrad shared memory buffer with the other smem buffers.
        //
        __shared__ uint4 smem_buf[spio::max(
            SmemInput::size + SmemDelta::size,
            static_cast<int>(SmemWgrad::num_bytes / sizeof(uint4)))];
        uint4 *smem_input_buf = smem_buf;
        uint4 *smem_delta_buf = smem_buf + SmemInput::size;
        float2 *smem_wgrad_buf = reinterpret_cast<float2 *>(smem_buf);

        //
        // Define the block tile.
        //
        BlockIdx block_idx(blockIdx.x);
        auto block_n = block_idx.block_n();
        auto block_y = block_idx.block_y();
        auto block_q = block_idx.block_q();
        auto block_c = block_idx.block_c();

        //
        // Define tile mappings.
        //

        // Load input to smem.
        Input global_input(input_ptr);
        SmemInput smem_input_store(smem_input_buf);
        bool thread_loads_input;
        bool input_inbounds;
        auto input_n = [&]()
        {
            InputIdx idx(threadIdx.x);
            smem_input_store = smem_input_store[idx.n()][idx.x()][idx.c8()];
            auto _input_n = idx.n();
            auto block_x = block_q.unfold().cast<X_Dim>() - PADDING_W;
            auto x = block_x + idx.x();
            auto c8 = block_c.fold<8>() + idx.c8();
            global_input = global_input[x][c8];
            thread_loads_input = threadIdx.x < InputIdx::size;
            bool x_inbounds = (x >= 0 && x < Input::X);
            bool c8_inbounds = (c8 < Input::C8);
            input_inbounds = (x_inbounds && c8_inbounds);
            return _input_n;
        }();

        // Load delta to smem.
        Delta global_delta(deltas_ptr);
        SmemDelta smem_delta_store(smem_delta_buf);
        bool delta_inbounds;
        bool thread_loads_delta;
        auto delta_n = [&]()
        {
            DeltaIdx idx(threadIdx.x);
            auto _delta_n = idx.n();
            smem_delta_store = smem_delta_store[idx.n()][idx.q()][idx.k8()];
            auto q = block_q.unfold() + idx.q();
            auto k8 = block_c.fold<8>().cast<K_Dim>() + idx.k8();
            global_delta = global_delta[q][k8];
            thread_loads_delta = threadIdx.x < DeltaIdx::size;
            delta_inbounds = (k8 < Delta::K8 && q < Delta::Q);
            return _delta_n;
        }();

        // Load input-smem to register.
        SmemInput smem_input_load(smem_input_buf);
        {
            SmemInputLoadIdx idx(threadIdx.x);
            smem_input_load = smem_input_load[idx.q().cast<X_Dim>() + idx.warp_s().unfold().cast<X_Dim>() + idx.s().cast<X_Dim>()][idx.c8()];
        }

        // Load delta-smem to register.
        SmemDelta smem_delta_load(smem_delta_buf);
        {
            SmemDeltaLoadIdx idx(threadIdx.x);
            smem_delta_load = smem_delta_load[idx.q()][idx.k8()];
        }

        //
        // Declare the accumulators.
        //
        AccTensor::data_type acc_array[AccTensor::size];
        AccTensor acc(acc_array);
        for (auto s2 : acc.S2)
        {
            for (auto r : acc.R)
            {
                acc.s2(s2).r(r)->zero();
            }
        }

        // Iterate over batches.
        for (int n_iter = 0; n_iter < BLOCK_N_ITERS; ++n_iter)
        {
            //
            // Define the pipeline.
            //
            Pipeline pipeline;
            constexpr unsigned STAGE_GLOBAL_DELTAS_LOAD = 1 << 0;
            constexpr unsigned STAGE_SMEM_DELTAS_LOAD = 1 << 1;
            constexpr unsigned STAGE_GLOBAL_INPUT_LOAD = 1 << (R - 1);
            constexpr unsigned STAGE_COMPUTE = 1 << R;
            constexpr int NUM_ITERS = R + BLOCK_Y_Dim(1).unfold().get();

            //
            // Define the input and delta (grad_output) pointers for the current batch iteration.
            //
            auto input_n_iter = block_n.unfold() + input_n + n_iter * WARP_N;
            bool input_n_inbounds = (input_n_iter < Input::N);
            auto global_input_n_iter = global_input.n(input_n_iter);

            auto delta_n_iter = block_n.unfold() + delta_n + n_iter * WARP_N;
            bool delta_n_inbounds = (delta_n_iter < Delta::N);
            auto global_delta_n_iter = global_delta.n(delta_n_iter);

            int y = block_y.unfold().get();
            int p = y - TRANSPOSE_PADDING_H;
            int ping_pong = 0;

            //
            // Define the deltas fragments.
            //
            DeltaTensor::data_type delta_array[DeltaTensor::size];
            DeltaTensor deltas(delta_array);

            // Run the pipeline, unrolling it R times.
            for (int iter = 0; iter < NUM_ITERS; iter += R)
            {
                for (int phase = 0; phase < R && iter + phase < NUM_ITERS; ++phase)
                {
                    pipeline.step(iter + phase < BLOCK_P_Dim(1).unfold().get());
                    if (pipeline.active(STAGE_GLOBAL_INPUT_LOAD, STAGE_GLOBAL_DELTAS_LOAD))
                    {
                        if (thread_loads_input)
                        {
                            memcpy_async(
                                smem_input_store.ping_pong(ping_pong).get(),
                                global_input_n_iter.y(y).get(),
                                input_inbounds && input_n_inbounds && (y >= 0 && y < Input::Y.get()));
                        }
                        ++y;
                    }
                    if (pipeline.active(STAGE_GLOBAL_DELTAS_LOAD))
                    {
                        if (thread_loads_delta)
                        {
                            memcpy_async(
                                smem_delta_store.ping_pong(ping_pong).get(),
                                global_delta_n_iter.p(p).get(),
                                delta_inbounds && delta_n_inbounds && (p >= 0 && p < Delta::P.get()));
                        }
                        __pipeline_commit();
                        ++p;
                    }
                    ping_pong ^= 1;
                    if (pipeline.active(STAGE_SMEM_DELTAS_LOAD))
                    {
                        __pipeline_wait_prior(pipeline.active(STAGE_GLOBAL_DELTAS_LOAD) ? 1 : 0);
                        __syncthreads();
                        for (auto warp_n : deltas.N)
                        {
                            int r_idx = (R - 1 + phase) % R;
                            deltas.n(warp_n).r(r_idx)->load_trans(smem_delta_load.ping_pong(ping_pong).n(warp_n).get());
                        }
                    }
                    if (pipeline.active(STAGE_COMPUTE))
                    {
                        for (auto warp_n : deltas.N)
                        {
                            InputTensor::data_type input_array[InputTensor::size];
                            InputTensor input(input_array);
                            for (auto s2 : input.S2)
                            {
                                input[s2]->load_trans(smem_input_load.ping_pong(ping_pong)[warp_n][s2.unfold().cast<X_Dim>()].get());
                            }
                            for (auto s2 : input.S2)
                            {
                                for (auto r : acc.R)
                                {
                                    mma_trans(*acc[s2][r], *input[s2], *deltas[warp_n][(r + phase) % R], *acc[s2][r]);
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

        // Store accumulator to wgrad-smem.
        auto global_wgrad = Wgrad(wgrad_ptr)[block_c.unfold().cast<K_Dim>()];
        SmemWgrad smem_wgrad_store(smem_wgrad_buf);
        auto warp_s = [&]()
        {
            SmemWgradStoreIdx idx(threadIdx.x);
            auto _warp_s = idx.warp_s().unfold();
            Acc::Index acc_idx(idx.lane());
            smem_wgrad_store = smem_wgrad_store[idx.k8()][acc_idx.k2()][_warp_s][acc_idx.c()];
            return _warp_s;
        }();

        // Note: Out-of-resources CUDA error when coded as a range-based loop (auto r : acc.R)
#pragma unroll R
        for (int r = 0; r < R; ++r)
        {
            if (r > 0)
            {
                __syncthreads();
            }
            for (auto s : WARP_S_Dim::stride)
            {
                if (warp_s + s >= S)
                {
                    break;
                }
                auto sd2 = s.fold<2>();
                auto sm2 = s % 2;
                *smem_wgrad_store[s] = acc[sd2][R_Dim(r)]->fragment(sm2.get());
            }
            __syncthreads();

            // Add wgrad to the global result.
            SmemWgrad smem_wgrad_load(smem_wgrad_buf);
#pragma unroll 1
            for (int iter = threadIdx.x; iter < WgradStoreIdx::size; iter += Block::threads)
            {
                // Flip r-dimension.
                WgradStoreIdx idx(iter);
                auto smem_wgrad_load_iter = smem_wgrad_load[idx.k8()][idx.s()][idx.c()];
                auto wgrad_iter = global_wgrad[idx.k8().unfold()][acc.R - 1 - r][idx.s()][idx.c()];
                auto k = block_c.unfold().cast<K_Dim>() + idx.k8().unfold();
#pragma unroll 4
                for (int k2 = 0; k2 < 4; ++k2)
                {
                    if (k + k2 * 2 < Wgrad::K)
                    {
                        float2 wgrad_f2 = *smem_wgrad_load_iter.k2(k2);
                        atomicAdd(wgrad_iter.k(k2 * 2 + 0).get(), wgrad_f2.x);
                        atomicAdd(wgrad_iter.k(k2 * 2 + 1).get(), wgrad_f2.y);
                    }
                }
            }
        }
    }
}