#include <cuda_pipeline.h>

#include "spio/fragment_mma.cuh"
#include "spio/ldmatrix.cuh"
#include "spio/fragment_layout.h"
#include "spio/mathutil.h"

#include "parameters.h"

using namespace spio;


extern "C"
{
    __global__ void SPIO_MLP_KERNEL(
        uint4 *__restrict__ output_ptr,
        const uint4 *__restrict__ input_ptr,
        const uint4 *__restrict__ exp_weights_ptr,
        const uint4 *__restrict__ exp_bias_ptr,
        const uint4 *__restrict__ prj_weights_ptr,
        const uint4 *__restrict__ prj_bias_ptr)
    {
        // Define the shared memory buffers.
        extern __shared__ uint4 smem[];
        uint4 *smem_input_ptr = smem;
        uint4 *smem_exp_weights_ptr = smem;
        uint4 *smem_prj_weights_ptr = smem + SmemExpWeight::size;

        int block_x = blockIdx.x * Block::x;

        WarpIdx warp_idx(threadIdx.x);

        // Define the input matrix fragments.
        //
        // The input tile stays resident in registers throughout the kernel.
        In::data_type in_array[In::size];
        In in(in_array);

        // Load the input tile.
        {
            // Load the input tile into shared memory.
            int global_x = block_x + warp_idx.warp() * Params::warp_x16 * 16;

            auto input = Input(input_ptr).x(global_x);
            auto smem_input = SmemInput(smem_input_ptr).warp_x(warp_idx.warp());

            constexpr int Lanes = 32;
            constexpr int InputLoads = SmemInput::X * SmemInput::C8;

            for (int idx = warp_idx.lane(); idx < InputLoads; idx += Lanes)
            {
                SmemInput::Index smem_idx(idx);
                int zfill = (global_x + smem_idx.x()) < Input::X ? 0 : 16;
                __pipeline_memcpy_async(
                    smem_input.x(smem_idx.x()).c8(smem_idx.c8()).get(),
                    input.x(smem_idx.x()).c8(smem_idx.c8()).get(),
                    16,
                    zfill);
            }
            __pipeline_commit();
        }
        {
            // Load the input matrix fragments from shared memory.

            // No block synchronization is needed here, because each warp is loading its own input tile.
            __pipeline_wait_prior(0);

            LdmatrixAIdx smem_input_lane_idx(warp_idx.lane());
            auto smem_input_load = SmemInput(smem_input_ptr).warp_x(warp_idx.warp()).x(smem_input_lane_idx.m()).c8(smem_input_lane_idx.k8());
            for (int c16 = 0; c16 < Params::c16; ++c16)
            {
                for (int x16 = 0; x16 < Params::warp_x16; ++x16)
                {
                    in.c16(c16).x16(x16)->reg4() = ldmatrix_x4(smem_input_load.x(x16 * 16).c8(c16 * 2).get());
                }
            }
        }

        // Release smem_input shared memory.
        __syncthreads();

        // Initialize the output accumulators.
        Out::data_type out_array[Out::size];
        Out out_acc(out_array);
        for (int k16 = 0; k16 < Out::K16; ++k16)
        {
            for (int x16 = 0; x16 < Out::X16; ++x16)
            {
                out_acc.k16(k16).x16(x16)->zero();
            }
        }

        CheckerboardGlobal global_lane_idx(warp_idx.lane());

        // Map the expansion weights to shared memory.
        constexpr int exp_max_warp_loads = divup(ExpWeight::C16, ExpWeightLoadIdx::WARP_C16);
        int exp_num_warp_loads = (ExpWeight::C16 - 1 - warp_idx.warp()) / ExpWeightLoadIdx::WARP_C16;
        ExpWeight exp_weight(exp_weights_ptr);
        SmemExpWeight smem_exp_weight_store(smem_exp_weights_ptr);
        {
            ExpWeightLoadIdx idx(warp_idx.warp());
            int lane_c8 = global_lane_idx.k8();
            int lane_r = global_lane_idx.m();
            int global_r = idx.warp_r16() * 16 + lane_r;
            exp_weight = exp_weight.c16(idx.warp_c16()).r(global_r).c8(lane_c8);
            smem_exp_weight_store = smem_exp_weight_store.r16(idx.warp_r16()).c16(idx.warp_c16()).checkerboard(global_lane_idx.offset<8>());
        }

        // Map the projection weights to shared memory.
        constexpr int prj_max_warp_loads = divup(PrjWeight::K, PrjWeightLoadIdx::WARP_K16 * 16);
        int prj_num_warp_loads = (PrjWeight::K / 16 - 1 - warp_idx.warp()) / PrjWeightLoadIdx::WARP_K16;
        PrjWeight prj_weight(prj_weights_ptr);
        SmemPrjWeight smem_prj_weight_store(smem_prj_weights_ptr);
        {
            PrjWeightLoadIdx idx(warp_idx.warp());
            int lane_r8 = global_lane_idx.k8();
            int lane_k = global_lane_idx.m();
            int global_k = idx.warp_k16() * 16 + lane_k;
            prj_weight = prj_weight.k(global_k).r16(idx.warp_r16()).r8(lane_r8);
            smem_prj_weight_store = smem_prj_weight_store.k16(idx.warp_k16()).r16(idx.warp_r16()).checkerboard(global_lane_idx.offset<8>());
        }

        CheckerboardB smem_weight_load_lane_idx(warp_idx.lane());
        auto smem_exp_weight_load = SmemExpWeight(smem_exp_weights_ptr).checkerboard(smem_weight_load_lane_idx.offset<8>());
        auto smem_prj_weight_load = SmemPrjWeight(smem_prj_weights_ptr).checkerboard(smem_weight_load_lane_idx.offset<8>());

        int ping_pong = 0;

        // Main loop over hidden layer chunks.
        for (int iter = 0; iter < Params::num_r_chunks + 1; ++iter)
        {
            int iter_r16 = iter * Params::r16_chunk;
            if (iter < Params::num_r_chunks)
            {
                // Load the expansion weights asynchronously for the next iteration.
                for (int load_idx = 0; load_idx < exp_num_warp_loads; ++load_idx)
                {
                    int iter_c16 = load_idx * ExpWeightLoadIdx::WARP_C16;
                    // TODO: bounds checking for c16 and r16.
                    __pipeline_memcpy_async(
                        smem_exp_weight_store.c16(iter_c16).ping_pong(ping_pong).get(),
                        exp_weight.c16(iter_c16).r(iter_r16 * 16).get(),
                        16);
                }

                // Load the projection weights asynchronously for the next iteration.
                for (int load_idx = 0; load_idx < prj_num_warp_loads; ++load_idx)
                {
                    int iter_k16 = load_idx * PrjWeightLoadIdx::WARP_K16;
                    // TODO: bounds checking for k16 and r16.
                    __pipeline_memcpy_async(
                        smem_prj_weight_store.k16(iter_k16).ping_pong(ping_pong).get(),
                        prj_weight.k(iter_k16 * 16).r16(iter_r16).get(),
                        16);
                }
                __pipeline_commit();
            }

            ping_pong = 1 - ping_pong;

            if (iter > 0)
            {
                __pipeline_wait_prior((iter < Params::num_r_chunks) ? 1 : 0);
                __syncthreads();

                // Initialize the accumulators for the hidden layer pre-activations.
                Hidden::data_type hidden_acc_array[Hidden::size];
                Hidden hidden_acc(hidden_acc_array);
                for (int r16 = 0; r16 < Hidden::R16; ++r16)
                {
                    for (int x16 = 0; x16 < Hidden::X16; ++x16)
                    {
                        hidden_acc.r16(r16).x16(x16)->zero();
                    }
                }

                // Compute the hidden layer pre-activations.
                for (int c16 = 0; c16 < In::C16; ++c16)
                {
                    for (int r16 = 0; r16 < Hidden::R16; ++r16)
                    {
                        MMA_N16_K16_F16_B wgt_exp;
                        wgt_exp.vector() = ldmatrix_x4(smem_exp_weight_load.ping_pong(ping_pong).r16(r16).c16(c16).get());
                        for (int x16 = 0; x16 < In::X16; ++x16)
                        {
                            matmul_trans(*hidden_acc.x16(x16).r16(r16), *in.c16(c16).x16(x16), wgt_exp, *hidden_acc.x16(x16).r16(r16));
                        }
                    }
                }

                // Compute the hidden layer activations.
                HiddenAct::data_type hidden_act_array[HiddenAct::size];
                HiddenAct hidden_act(hidden_act_array);
                for (int r16 = 0; r16 < Hidden::R16; ++r16)
                {
                    for (int x16 = 0; x16 < Hidden::X16; ++x16)
                    {
                        for (int idx = 0; idx < 4; ++idx)
                        {
                            hidden_act.r16(r16).x16(x16)->fragment(idx) = hidden_acc.r16(r16).x16(x16)->to_half2(idx);
                        }
                    }
                }

                // Compute the projection.
                for (int r16 = 0; r16 < HiddenAct::R16; ++r16)
                {
                    for (int k16 = 0; k16 < Params::k16; ++k16)
                    {
                        MMA_N16_K16_F16_B wgt_prj;
                        wgt_prj.vector() = ldmatrix_x4(smem_prj_weight_load.ping_pong(ping_pong).k16(k16).r16(16).get());
                        for (int x16 = 0; x16 < HiddenAct::X16; ++x16)
                        {
                            matmul_trans(*out_acc.k16(k16).x16(x16), *hidden_act.r16(r16).x16(x16), wgt_prj, *out_acc.k16(k16).x16(x16));
                        }
                    }
                }

                __syncthreads();
            }
        }

        // Store the outputs to shared memory.
        SmemOutputStore smem_output_store(reinterpret_cast<__half2 *>(smem));
        {
            int warp_x16 = warp_idx.warp() * Params::warp_x16;
            int warp_x8 = warp_x16 * 2;
            smem_output_store = smem_output_store.x8(warp_x8).lane(warp_idx.lane());
        }

        for (int x16 = 0; x16 < HiddenAct::X16; ++x16)
        {
            for (int k16 = 0; k16 < Params::k16; ++k16)
            {
                // TODO: Use a FragmentSpec  to define the output_acc fragment type.
                // TODO: Use Out::data_type::Index to compute x8() and k2(). 
                for (int k8 = 0; k8 < 2; ++k8)
                {
                    for (int x8 = 0; x8 < 2; ++x8)
                    {
                        *smem_output_store.x8(x16 * 2 + x8).k8(k16 * 2 + k8) = out_acc.k16(k16).x16(x16)->to_half2(k8 * 2 + x8);
                    }
                }
            }
        }

        // Each warp's output writes are independent, so no block synchronization is needed.
        __syncwarp();

        // Transfer the outputs from shared memory to global memory.
        Output output(output_ptr);
        int warp_x8 = warp_idx.warp() * Params::warp_x16 * 2;
        int global_x_warp = block_x + warp_x8 * 8;
        auto smem_output_load = SmemOutputLoad(smem).x8(warp_x8);

        for (int iter = warp_idx.lane(); iter < SmemOutputLoad::size; iter += Block::threads)
        {
            SmemOutputLoadIdx idx(iter);
            int global_x = global_x_warp + idx.x8() * 8 + idx.xm8();
            if (global_x < Output::X && idx.k8() < Output::K8)
            {
                *output.x(global_x).k8(idx.k8()) = *smem_output_load.x8(idx.x8()).k8(idx.k8()).xm8(idx.xm8());
            }
        }
    }
}

