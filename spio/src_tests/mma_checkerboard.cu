#include <cuda_pipeline_primitives.h>

#include "spio/fragment_mma.cuh"
#include "spio/ldmatrix.cuh"
#include "spio/fragment_load_index.h"
#include "spio/fragment_index.h"
#include "spio/checkerboard_index.h"
#include "spio/mathutil.h"

#include "parameters.h"

extern "C"
{
    using namespace spio;

    /// @brief Test matrix multiplication with checkerboard layout.
    ///
    /// This kernel uses the checkerboard layout for shared memory when
    /// loading the A and B matrices.
    ///
    /// The checkerboard is a 16x2 grid with a vector
    /// of 8 half2 elements per cell. See checkerboard.h for details.
    ///
    /// @param c_ptr result matrix with float32 precision.
    /// @param a_ptr operand A matrix with float16 precision.
    /// @param b_ptr operand B matrix with float16 precision.
    __global__ void mma_checkerboard(
        uint4 *__restrict__ c_ptr,
        const uint4 *__restrict__ a_ptr,
        const uint4 *__restrict__ b_ptr)
    {
        __shared__ uint4 smem[spio::max(SmemA::size + SmemB::size, SmemCLoad::size)];

        uint4 *smem_a = smem + 0;
        uint4 *smem_b = smem + SmemA::size;

        BLOCK_I_Dim block_i(blockIdx.x);
        BLOCK_J_Dim block_j(blockIdx.y);

        // The lanes of a warp load 16x 2k [8k] [half2] from global memory.
        GlobalLoadIndex global_load_idx(threadIdx.x);

        // Position the global memory input tile.
        auto a = A(a_ptr).i(block_i.unfold() + global_load_idx.x().cast<I_Dim>()).k8(global_load_idx.k8());
        auto b = B(b_ptr).j(block_j.unfold() + global_load_idx.x().cast<J_Dim>()).k8(global_load_idx.k8());

        // Define the mapping from the global memory tile to the shared memory tile.
        auto smem_a_store = SmemA(smem_a).checkers(global_load_idx.x().cast<I_Dim>(), global_load_idx.k8());
        auto smem_b_store = SmemB(smem_b).checkers(global_load_idx.x().cast<J_Dim>(), global_load_idx.k8());

        // Define the mapping from the thread index to the warps' i32 x j64 tiles and the lane index.
        ComputeIndex compute_idx(threadIdx.x);

        // Define the mapping from the shared memory tile to the register tile.
        A_Fragment::LoadIndex a_load_idx(compute_idx.lane());
        B_Fragment::LoadIndex b_load_idx(compute_idx.lane());

        auto smem_a_load = SmemA(smem_a).i32(compute_idx.i32()).checkers(a_load_idx.i(), a_load_idx.k8());
        auto smem_b_load = SmemB(smem_b).j64(compute_idx.j64()).checkers(b_load_idx.j(), b_load_idx.k8());

        int ping_pong = 0;

        // Initialize the accumulator.
        C_Tensor::data_type c_data[C_Tensor::size];
        C_Tensor c_tensor(c_data);

        for (auto i16 : C_Tensor::I16)
        {
            for (auto j16 : C_Tensor::J16)
            {
                c_tensor.i16(i16).j16(j16)->zero();
            }
        }

        // Iterate over 16-element chunks of the k-dimension.
        for (int iter = 0; iter < Params::k16 + 1; ++iter)
        {
            if (iter < Params::k16)
            {
                __pipeline_memcpy_async(smem_a_store.ping_pong(ping_pong).get(), a.get(), Params::vec_bytes);
                __pipeline_memcpy_async(smem_b_store.ping_pong(ping_pong).get(), b.get(), Params::vec_bytes);
                __pipeline_commit();
                a = a.k8(2);
                b = b.k8(2);
            }

            ping_pong ^= 1;

            if (iter > 0)
            {
                __pipeline_wait_prior(iter < Params::k16 ? 1 : 0);

                __syncthreads();

                A_Tensor::data_type a_data[A_Tensor::size];
                A_Tensor a_tensor(a_data);
                for (auto i16 : A_Tensor::I16)
                {
                    a_tensor.i16(i16)->load(smem_a_load.ping_pong(ping_pong).i16(i16).get());
                }

                B_Tensor::data_type b_data[B_Tensor::size];
                B_Tensor b_tensor(b_data);
                for (auto j16 : B_Tensor::J16)
                {
                    b_tensor.j16(j16)->load(smem_b_load.ping_pong(ping_pong).j16(j16).get());
                }

                for (auto i16 : A_Tensor::I16)
                {
                    for (auto j16 : B_Tensor::J16)
                    {
                        mma_trans(*c_tensor.i16(i16).j16(j16), *a_tensor.i16(i16), *b_tensor.j16(j16), *c_tensor.i16(i16).j16(j16));
                    }
                }

                __syncthreads();
            }
        }

        // Store outputs through shared memory.
        C_Fragment::Index c_idx(compute_idx.lane());
        auto smem_c_store = SmemCStore(reinterpret_cast<__half2 *>(smem)).i32(compute_idx.i32()).j64(compute_idx.j64()).j2(c_idx.j2m4());
        for (auto i16 : C_Tensor::I16)
        {
            for (auto j16 : C_Tensor::J16)
            {
                for (int f = 0; f < C_Fragment::size(); ++f)
                {
                    *smem_c_store.j16(j16).j8(c_idx.j8(f)).i16(i16).i(c_idx.i(f)) = c_tensor.i16(i16).j16(j16)->to_half2(f);
                }
            }
        }

        // Transfer outputs from shared memory to global memory.
        auto c = C(c_ptr).i(block_i.unfold() + compute_idx.i32().unfold()).j8(block_j.fold<8>() + compute_idx.j64().fold<8>());
        auto smem_c_load = SmemCLoad(reinterpret_cast<const uint4 *>(smem)).i32(compute_idx.i32()).j64(compute_idx.j64());
        for (int offset = compute_idx.lane().get(); offset < SmemCLoadIndex::size; offset += ComputeIndex::LANE.get())
        {
            SmemCLoadIndex idx(offset);
            *c.i(idx.i()).j8(idx.j8()) = *smem_c_load.j8(idx.j8()).i(idx.i());
        }
    }
}