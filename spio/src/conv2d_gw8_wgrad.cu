#include <cuda_pipeline.h>
#include <cuda_fp16.h>

#include "spio/pipeline.h"
#include "spio/mma.cuh"
#include "spio/ldmatrix.cuh"

#include "parameters.h"

using namespace spio;
using namespace Params;

namespace
{
    __device__ constexpr int _max(int a, int b)
    {
        return a > b ? a : b;
    }
}

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
        __shared__ uint4 smem_buf[_max(SmemInput::size + SmemDelta::size, SmemWgrad::num_bytes / sizeof(uint4))];
        uint4 *smem_input_buf = smem_buf;
        uint4 *smem_delta_buf = smem_buf + SmemInput::size;
        float2 *smem_wgrad_buf = reinterpret_cast<float2 *>(smem_buf);

        //
        // Define the block tile.
        //
        BlockIdx block_idx(blockIdx.x);
        int block_n = block_idx.n() * Block::n;
        int block_y = block_idx.y() * Block::h;
        int block_q = block_idx.q() * Block::q;
        int block_c8 = block_idx.c8() * Block::c8;
        int block_c = block_c8 * 8;

        //
        // Define tile mappings.
        //

        // Load input to smem.
        Input global_input(input_ptr);
        SmemInput smem_input_store(smem_input_buf);
        bool thread_loads_input;
        int input_zfill;
        int input_n;
        {
            InputIdx idx(threadIdx.x);
            smem_input_store = smem_input_store.n(idx.n()).x(idx.x()).c8(idx.c8());
            input_n = idx.n();
            int block_x = block_q - PADDING_W;
            int x = block_x + idx.x();
            int c8 = block_c8 + idx.c8();
            global_input = global_input.x(x).c8(c8);
            thread_loads_input = threadIdx.x < InputIdx::size;
            bool x_inbounds = (x >= 0 && x < Input::X);
            bool c8_inbounds = (c8 < Input::C8);
            bool thread_inbounds = (x_inbounds && c8_inbounds);
            input_zfill = thread_inbounds ? 0 : sizeof(Input::data_type);
        }

        // Load delta to smem.
        Delta global_delta(deltas_ptr);
        SmemDelta smem_delta_store(smem_delta_buf);
        int delta_zfill;
        int delta_n;
        bool thread_loads_delta;
        {
            DeltaIdx idx(threadIdx.x);
            delta_n = idx.n();
            smem_delta_store = smem_delta_store.n(idx.n()).q(idx.q()).k8(idx.k8());
            int q = block_q + idx.q();
            int k8 = block_c8 + idx.k8();
            global_delta = global_delta.q(q).k8(k8);
            thread_loads_delta = threadIdx.x < DeltaIdx::size;
            bool delta_inbounds = (k8 < Delta::K8 && q < Delta::Q);
            delta_zfill = delta_inbounds ? 0 : sizeof(Delta::data_type);
        }

        // Load input-smem to register.
        SmemInput smem_input_load(smem_input_buf);
        {
            SmemInputLoadIdx idx(threadIdx.x);
            int warp_s = idx.warp_s() * WARP_S;
            smem_input_load = smem_input_load.x(idx.q() + warp_s + idx.s()).c8(idx.c8());
        }

        // Load delta-smem to register.
        SmemDelta smem_delta_load(smem_delta_buf);
        {
            SmemDeltaLoadIdx idx(threadIdx.x);
            smem_delta_load = smem_delta_load.q(idx.q()).k8(idx.k8());
        }

        //
        // Declare the accumulators.
        //
        Acc acc_array[AccTensor::size];
        AccTensor acc(acc_array);
        for (int s2 = 0; s2 < WARP_S2_UP; ++s2)
        {
            for (int r = 0; r < R; ++r)
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
            constexpr int NUM_ITERS = R + Block::h;

            //
            // Define the input and delta (grad_output) pointers for the current batch iteration.
            //
            int input_n_iter = block_n + input_n + n_iter * WARP_N;
            bool input_n_inbounds = (input_n_iter < Input::N);
            auto global_input_n_iter = global_input.n(input_n_iter);

            int delta_n_iter = block_n + delta_n + n_iter * WARP_N;
            bool delta_n_inbounds = (delta_n_iter < Delta::N);
            auto global_delta_n_iter = global_delta.n(delta_n_iter);

            int y = block_y;
            int p = y - TRANSPOSE_PADDING_H;
            int ping_pong = 0;

            //
            // Define the deltas fragments.
            //
            MMA_N8_K8_F16_B delta_array[DeltaFrag::size];
            DeltaFrag deltas(delta_array);

            // Run the pipeline, unrolling it R times.
            for (int iter = 0; iter < NUM_ITERS; iter += R)
            {
                for (int phase = 0; phase < R && iter + phase < NUM_ITERS; ++phase)
                {
                    pipeline.step(iter + phase < Block::p);
                    if (pipeline.active(STAGE_GLOBAL_INPUT_LOAD) && pipeline.active(STAGE_GLOBAL_DELTAS_LOAD))
                    {
                        bool y_inbounds = (y >= 0 && y < Input::Y);
                        int ny_fill = (input_n_inbounds && y_inbounds) ? 0 : sizeof(Input::data_type);
                        if (thread_loads_input)
                        {
                            __pipeline_memcpy_async(smem_input_store.ping_pong(ping_pong).get(),
                                                    global_input_n_iter.y(y).get(),
                                                    sizeof(Input::data_type),
                                                    input_zfill | ny_fill);
                        }
                        ++y;
                    }
                    if (pipeline.active(STAGE_GLOBAL_DELTAS_LOAD))
                    {
                        bool p_inbounds = (p >= 0 && p < Delta::P);
                        int np_fill = (p_inbounds && delta_n_inbounds) ? 0 : sizeof(Delta::data_type);
                        if (thread_loads_delta)
                        {

                            __pipeline_memcpy_async(smem_delta_store.ping_pong(ping_pong).get(),
                                                    global_delta_n_iter.p(p).get(),
                                                    sizeof(Delta::data_type),
                                                    delta_zfill | np_fill);
                        }
                        __pipeline_commit();
                        ++p;
                    }
                    ping_pong = 1 - ping_pong;
                    if (pipeline.active(STAGE_SMEM_DELTAS_LOAD))
                    {
                        __pipeline_wait_prior(pipeline.active(STAGE_GLOBAL_DELTAS_LOAD) ? 1 : 0);
                        __syncthreads();
                        for (int warp_n = 0; warp_n < WARP_N; ++warp_n)
                        {
                            int r_idx = (R - 1 + phase) % R;
                            deltas.n(warp_n).r(r_idx)->reg() = ldmatrix_x1_trans(smem_delta_load.ping_pong(ping_pong).n(warp_n).get());
                        }
                    }
                    if (pipeline.active(STAGE_COMPUTE))
                    {
                        for (int warp_n = 0; warp_n < WARP_N; ++warp_n)
                        {
                            MMA_M16_K8_F16_A input[WARP_S2_UP];
                            for (int s2 = 0; s2 < WARP_S2_UP; ++s2)
                            {
                                input[s2].vector() = ldmatrix_x2_trans(smem_input_load.ping_pong(ping_pong).n(warp_n).x(s2 * 2).get());
                            }
                            for (int s2 = 0; s2 < WARP_S2_UP; ++s2)
                            {
                                for (int r = 0; r < R; ++r)
                                {
                                    int r_idx = (r + phase) % R;
                                    mma_m16_n8_k8(acc.s2(s2).r(r)->vec4(), input[s2].reg2(), deltas.n(warp_n).r(r_idx)->reg(), acc.s2(s2).r(r)->vec4());
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
        auto global_wgrad = Wgrad(wgrad_ptr).k(block_c);
        SmemWgrad smem_wgrad_store(smem_wgrad_buf);
        int warp_s;
        {
            SmemWgradStoreIdx idx(threadIdx.x);
            warp_s = idx.warp_s() * WARP_S;
            int lane_c = Acc::c(idx.lane(), 0);
            int lane_k2 = Acc::k2(idx.lane());
            smem_wgrad_store = smem_wgrad_store.k8(idx.k8()).k2(lane_k2).s(warp_s).c(lane_c);
        }

#pragma unroll R
        for (int r = 0; r < R; ++r)
        {
            if (r > 0)
            {
                __syncthreads();
            }
            for (int s = 0; s < WARP_S; ++s)
            {
                if (warp_s + s >= S)
                {
                    break;
                }
                int sd2 = s / 2;
                int sm2 = s % 2;
                *smem_wgrad_store.s(s) = acc.s2(sd2).r(r)->fragment(sm2);
            }
            __syncthreads();

            // Add wgrad to the global result.
            SmemWgrad smem_wgrad_load(smem_wgrad_buf);
#pragma unroll 1
            for (int iter = threadIdx.x; iter < WgradStoreIdx::size; iter += Block::threads)
            {
                // Flip r-dimension.
                WgradStoreIdx idx(iter);
                auto smem_wgrad_load_iter = smem_wgrad_load.k8(idx.k8()).s(idx.s()).c(idx.c());
                auto wgrad_iter = global_wgrad.k(idx.k8() * 8).r(R - 1 - r).s(idx.s()).c(idx.c());
                int k = block_c + idx.k8() * 8;
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