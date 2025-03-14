#include "spio.cuh"

#include "spio/fifo.cuh"
#include "spio/allocator.h"
#include "parameters.h"

using namespace spio;

namespace
{
    __device__ unsigned get_warp_id()
    {
        unsigned warp_id;
        asm("mov.u32 %0, %warpid;" : "=r"(warp_id));
        return warp_id;
    }

    __device__ unsigned get_lane()
    {
        unsigned lane_id;
        asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
        return lane_id;
    }

    __device__ unsigned get_smsp()
    {
        return warp_id() & 3;
    }

    __device__ unsigned get_warpgroup()
    {
        return (warp_id() >> 2) & 3;
    }
}

namespace
{
    constexpr int Lanes = 32;
}

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
        StackAllocator smem_alloc(smem);

        using InputBufferGuard = WarpFifoGuard<Params::warps_per_smsp>;

        auto warp = threadIdx / 32;
        auto lane = get_lane();

        auto warpgroup = get_warpgroup();
        auto smsp = get_smsp();

        auto partition_tid = warpgroup * 32 + lane;

        auto smem_exp_weights = SmemExpWeight::allocate(smem_alloc);
        auto smem_prj_weights = SmemPrjWeight::allocate(smem_alloc);
        auto smem_input = SmemInput::allocate(smem_alloc);
        auto input_buffer_fifo = InputBufferGuard::Fifo::allocate_resource_queue(
            smem_alloc,
            partition_tid,
            Params::num_smem_input_buf_per_partition);

        BLOCK_X_Dim block_x(blockIdx.x);

        // Load the expansion weights from global memory to shared memory.
        for (int offset = threadIdx.x; offset < ExpWeight::size; offset += blockDim.x)
        {
            ExpWeight exp_weight(exp_weights_ptr);
            ExpWeight::Index idx(offset);
            memcpy_async(
                smem_exp_weights[idx.c16()][idx.r().fold<16>()].checkers(idx.r() % 16, idx.c8()).get(),
                exp_weight.offset(offset).get())
        }
        __pipeline_commit();

        // Load the projection weights from global memory to shared memory.
        for (int offset = threadIdx.x; offset < PrjWeight::size, offset += blockDim.x)
        {
            PrjWeight prj_weight(prj_weights_ptr);
            PrjWeight::Index idx(offset);
            memcpy_async(
                smem_prj_weights[idx.r16()][idx.k().fold<16>()].checkers(idx.k() % 16, idx.r8()).get(),
                prj_weight.offset(offset).get())
        }
        __pipeline_commit();

        // Load the warp's input tile.
        In::data_type in_array[In::size];
        In in(in_array);
        {
            InputBufferGuard buffer_guard(input_buffer_fifo);
            auto smem_input_buf = smem_input.buffer(buffer_guard.value());

            // Load the input tile into shared memory.
            {
                WARP_X_Dim warp_x(warp);
                auto global_x = block_x.unfold() + warp_x.unfold();

                auto input = Input(input_ptr)[global_x];

                for (auto offset = lane; offset < SmemInputIdx::size; offset += Lanes)
                {
                    SmemInputIdx smem_idx(offset);
                    memcpy_async(
                        smem_input_buf.get(),
                        input.offset(offset).get(),
                        global_x + smem_idx.x() < Input::X);
                }
                __pipeline_commit();
            }
            // Load the input matrix fragments from shared memory.
            {
                __pipeline_wait_prior(0);

                In::data_type::LoadIndex in_load_idx(lane);
                auto smem_input_load = smem_input_buf.warp_x(warp)[in_load_idx.x()][in_load_idx.c8()];
                for (auto c16 : in.C16)
                {
                    for (auto x16 : in.X16)
                    {
                        in[c16][x16].load(smem_input_load[x16.unfold()][c16.fold<8>()].get());
                    }
                }
            }
        }

        // Initialize the output accumulators.
        Out::data_type out_array[Out::size];
        Out out_acc(out_array);
        for (auto k16 = out_acc.K16)
        {
            for (auto x16 : out_acc.X16)
            {
                out_acc[k16][x16]->zero();
            }
        }

        // Initialize the load address for the expansion weights.
        Exp::data_type::LoadIndex exp_load_idx(lane);
        auto smem_exp_load = smem_exp_weights.checkers(exp_load_idx.r(), exp_load_idx.c8());

        // Initialize the load address for the projection weights.
        Prj::data_type::LoadIndex prj_load_idx(lane);
        auto smem_prj_load = smem_prj_weights.checkers(proj_load_idx.k(), proj_load_idx.r8());


        // Main loop over hidden layer chunks.
        for (int iter = 0; iter < Params::num_r_chunks; ++iter)
        {
            int iter_r16 = iter * Params::r16_chunk;

            // Initialize the accumulators for the hidden layer pre-activations.
            Hidden::data_type hidden_acc_array[Hidden::size];
            Hidden hidden_acc(hidden_acc_array);
            for (auto r16 : hidden_acc.R16)
            {
                for (auto x16 : hidden_acc.X16)
                {
                    hidden_acc[r16][x16]->zero();
                }
            }

            // Compute the hidden layer pre-activations.
            for (auto c16 : in.C16)
            {
                for (auto r16: hidden_acc.R16)
                {
                    auto wgt_exp = Exp::from_smem(smem_exp_weight_load[r16][c16].get());
                    for (auto x16: in.X16)
                    {
                        matmul_trans(*hidden_acc[x16][r16], *in[c16][x16], wgt_exp, *hidden_acc[x16][r16]);
                    }
                }
            }

            // Compute the hidden layer activations.
            HiddenAct::data_type hidden_act_array[HiddenAct::size];
            HiddenAct hidden_act(hidden_act_array);
            for (auto r16 : hidden_act.R16)
            {
                for (auto x16 : hidden_act.X16)
                {
                    for (int idx = 0; idx < HiddenAct::data_type::size(); ++idx)
                    {
                        hidden_act[16][x16]->fragment(idx) = hidden_acc[r16][x16]->to_half2(idx);
                    }
                }
            }

            // Compute the projection.
            for (auto r16 : hidden_act.R16)
            {
                for (auto k16 : out_acc.K16)
                {
                    auto wgt_prj = Prj::from_smem(smem_prj_weight_load[k16][r16].get());
                    for (auto x16 : out_acc.X16)
                    {
                        matmul_trans(*out_acc[k16][x16], hidden_act[r16][x16], wgt_prj, *out_acc[k16][x16]);
                    }
                }
            }
        }

        // Store the outputs to shared memory
        // TODO define SmemOutputStore and SmemOutputLoad tensors.
        SmemOutputStore smem_output_store(reinterpret_cast<__half2 *>(smem));
        {
            int warp_x16 = warp_idx.warp() * Params::warp_x16;
            int warp_x8 = warp_x16 * 2;
            smem_output_store = smem_output_store.x8(warp_x8).lane(warp_idx.lane());
        }

        for (auto x16: out_acc.X16)
        {
            for (auto k16 : out_acc.K16)
            {
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
