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
    __device__ constexpr int divup(int n, int d)
    {
        return (n + d - 1) / d;
    }

    __device__ constexpr int _max(int a, int b)
    {
        return a > b ? a : b;
    }
}

extern "C"
{
    __global__ void spio_conv2d_gw8_wgrad(
        float *__restrict__ wgrad_ptr,
        const uint4 *__restrict__ input_ptr,
        const uint4 *__restrict__ deltas_ptr)
    {
        //
        // Define some constansts.
        //
        constexpr int S2_UP = divup(Wgrad::S, 2);
        constexpr int S2 = S / 2;

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
        int block_n = block_idx.n();
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
        {
            InputIdx idx(threadIdx.x);
            smem_input_store = smem_input_store.x(idx.x()).c8(idx.c8());
            int block_x = block_q - PADDING_W;
            int x = block_x + idx.x();
            int c8 = block_c8 + idx.c8();
            global_input = global_input.n(block_n).x(x).c8(c8);
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
        bool thread_loads_delta;
        {
            DeltaIdx idx(threadIdx.x);
            smem_delta_store = smem_delta_store.q(idx.q()).k8(idx.k8());
            int q = block_q + idx.q();
            int k8 = block_c8 + idx.k8();
            global_delta = global_delta.n(block_n).q(q).k8(k8);
            thread_loads_delta = threadIdx.x < DeltaIdx::size;
            bool delta_inbounds = (k8 < Delta::K8 && q < Delta::Q);
            delta_zfill = delta_inbounds ? 0 : sizeof(Delta::data_type);
        }

        // Load input-smem to register.
        SmemInput smem_input_load(smem_input_buf);
        {
            SmemInputLoadIdx idx(threadIdx.x);
            smem_input_load = smem_input_load.x(idx.q() + idx.s()).c8(idx.c8());
        }

        // Load delta-smem to register.
        SmemDelta smem_delta_load(smem_delta_buf);
        {
            SmemDeltaLoadIdx idx(threadIdx.x);
            smem_delta_load = smem_delta_load.q(idx.q()).k8(idx.k8());
        }

        // Store accumulator to wgrad-smem.
        SmemWgrad smem_wgrad_store(smem_wgrad_buf);
        {
            SmemWgradStoreIdx idx(threadIdx.x);
            int lane_c = Acc::c(idx.lane(), 0);
            int lane_k2 = Acc::k2(idx.lane());
            smem_wgrad_store = smem_wgrad_store.k8(idx.k8()).k2(lane_k2).c(lane_c);
        }

        //
        // Define the deltas fragments.
        //
        MMA_N8_K8_F16_B deltas[R];

        //
        int y = block_y;
        int p = y - TRANSPOSE_PADDING_H;
        int ping_pong = 0;
        Pipeline pipeline;
        constexpr unsigned STAGE_GLOBAL_DELTAS_LOAD = 1 << 0;
        constexpr unsigned STAGE_SMEM_DELTAS_LOAD = 1 << 1;
        constexpr unsigned STAGE_GLOBAL_INPUT_LOAD = 1 << (R - 1);
        constexpr unsigned STAGE_COMPUTE = 1 << R;
        constexpr int NUM_ITERS = R + Block::h;

        //
        // Declare the accumulators.
        //
        Acc acc_array[AccTensor::size];
        AccTensor acc(acc_array);
        for (int s2 = 0; s2 < S2_UP; ++s2)
        {
            for (int r = 0; r < R; ++r)
            {
                acc.s2(s2).r(r)->zero();
            }
        }

        // Run the pipeline, unrolling+ it R times.
        for (int iter = 0; iter < NUM_ITERS; iter += R)
        {
            for (int phase = 0; phase < R && iter + phase < NUM_ITERS; ++phase)
            {
                pipeline.step(iter + phase < Block::p);
                if (pipeline.active(STAGE_GLOBAL_INPUT_LOAD) && pipeline.active(STAGE_GLOBAL_DELTAS_LOAD))
                {
                    bool y_inbounds = (y >= 0 && y < Input::Y);
                    int y_fill = y_inbounds ? 0 : sizeof(Input::data_type);
                    if (thread_loads_input)
                    {
                        __pipeline_memcpy_async(smem_input_store.ping_pong(ping_pong).get(),
                                                global_input.y(y).get(),
                                                sizeof(Input::data_type),
                                                input_zfill | y_fill);
                    }
                    ++y;
                }
                if (pipeline.active(STAGE_GLOBAL_DELTAS_LOAD))
                {
                    bool p_inbounds = (p >= 0 && p < Delta::P);
                    int p_fill = p_inbounds ? 0 : sizeof(Delta::data_type);
                    if (thread_loads_delta)
                    {

                        __pipeline_memcpy_async(smem_delta_store.ping_pong(ping_pong).get(),
                                                global_delta.p(p).get(),
                                                sizeof(Delta::data_type),
                                                delta_zfill | p_fill);
                    }
                    __pipeline_commit();
                    ++p;
                }
                ping_pong = 1 - ping_pong;
                if (pipeline.active(STAGE_SMEM_DELTAS_LOAD))
                {
                    __pipeline_wait_prior(pipeline.active(STAGE_GLOBAL_DELTAS_LOAD) ? 1 : 0);
                    __syncthreads();
                    deltas[(R - 1 + phase) % R].reg() = ldmatrix_x1_trans(smem_delta_load.ping_pong(ping_pong).get());
                }
                if (pipeline.active(STAGE_COMPUTE))
                {
                    MMA_M16_K8_F16_A input[S2_UP];
                    for (int s2 = 0; s2 < S2; ++s2)
                    {
                        input[s2].vector() = ldmatrix_x2_trans(smem_input_load.ping_pong(ping_pong).x(s2 * 2).get());
                    }
                    if constexpr (S2 < S2_UP)
                    {
                        input[S2_UP - 1].reg(0) = ldmatrix_x1_trans(smem_input_load.ping_pong(ping_pong).x(S - 1).get());
                        input[S2_UP - 1].reg(1) = 0;
                    }
                    for (int s2 = 0; s2 < S2_UP; ++s2)
                    {
                        for (int r = 0; r < R; ++r)
                        {
                            mma_m16_n8_k8(acc.s2(s2).r(r)->vec4(), input[s2].reg2(), deltas[(r + phase) % R].reg(), acc.s2(s2).r(r)->vec4());
                        }
                    }
                }
                if (pipeline.active(STAGE_SMEM_DELTAS_LOAD))
                {
                    __syncthreads();
                }
            }
        }

        // Store accumulators to wgrad-smem.
        for (int r = 0; r < R; ++r)
        {
            if (r > 0)
            {
                __syncthreads();
            }
            for (int s = 0; s < S; ++s)
            {
                int sd2 = s / 2;
                int sm2 = s % 2;
                // Flip r-dimension.
                *smem_wgrad_store.s(s) = acc.s2(sd2).r(r)->fragment(sm2);
            }
            __syncthreads();

            // Add wgrad to the global result.
            auto global_wgrad = Wgrad(wgrad_ptr).k(block_c);
            SmemWgrad smem_wgrad_load(smem_wgrad_buf);
            for (int iter = threadIdx.x; iter < WgradStoreIdx::size; iter += Block::threads)
            {
                WgradStoreIdx idx(iter);
                auto smem_wgrad_load_iter = smem_wgrad_load.k8(idx.k8()).s(idx.s()).c(idx.c());
                auto wgrad_iter = global_wgrad.k(idx.k8() * 8).r(R - 1 - r).s(idx.s()).c(idx.c());
                for (int k2 = 0; k2 < 4; ++k2)
                {
                    float2 wgrad_f2 = *smem_wgrad_load_iter.k2(k2);
                    atomicAdd(wgrad_iter.k(k2 * 2 + 0).get(), wgrad_f2.x);
                    atomicAdd(wgrad_iter.k(k2 * 2 + 1).get(), wgrad_f2.y);
                }
            }
        }
    }
}