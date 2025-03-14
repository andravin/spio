#include "spio.cuh"

#include "spio/allocator.h"

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
    /// @param c_ptr result matrix with float16 precision.
    /// @param a_ptr operand A matrix with float16 precision.
    /// @param b_ptr operand B matrix with float16 precision.
    __global__ void
    mma_checkerboard_16c(
        uint4 *__restrict__ c_ptr,
        const uint4 *__restrict__ a_ptr,
        const uint4 *__restrict__ b_ptr)
    {
        __shared__ uint4 smem[spio::max(SmemA::size + SmemB::size, SmemCLoad::size)];

        StackAllocator smem_allocator(smem);
        auto smem_a = SmemA::allocate(smem_allocator);
        auto smem_b = SmemB::allocate(smem_allocator);

        BLOCK_I_Dim block_i(blockIdx.x);
        BLOCK_J_Dim block_j(blockIdx.y);

        // The lanes of a warp load 16x 2k [8k] [half2] from global memory.
        GlobalLoadIndex global_load_idx(threadIdx.x);

        // Position the global memory input tiles.
        auto a = A(a_ptr)[block_i.unfold() + global_load_idx.x().cast<I_Dim>()][global_load_idx.k8()];
        auto b = B(b_ptr)[block_j.unfold() + global_load_idx.x().cast<J_Dim>()][global_load_idx.k8()];

        // Define the mapping from the global memory tile to the shared memory tile.
        auto smem_a_store = SmemA(smem_a).checkers(global_load_idx.x().cast<I_Dim>(), global_load_idx.k8());
        auto smem_b_store = SmemB(smem_b).checkers(global_load_idx.x().cast<J_Dim>(), global_load_idx.k8());

        // Define the mapping from the thread index to the warps' i32 x j64 tiles and the lane index.
        ComputeIndex compute_idx(threadIdx.x);

        // Define the mapping from the shared memory tile to the register tile.
        A_Fragments::data_type::LoadIndex a_load_idx(compute_idx.lane());
        B_Fragments::data_type::LoadIndex b_load_idx(compute_idx.lane());
        auto smem_a_load = SmemA(smem_a)[compute_idx.i32()].checkers(a_load_idx.i(), a_load_idx.k8());
        auto smem_b_load = SmemB(smem_b)[compute_idx.j64()].checkers(b_load_idx.j(), b_load_idx.k8());

        // Double-buffer the global memory loads.
        int ping_pong = 0;

        // Initialize the accumulator.
        C_Fragments::data_type c_data[C_Fragments::size];
        C_Fragments c_tensor(c_data);

        for (auto i16 : c_tensor.I16)
        {
            for (auto j16 : c_tensor.J16)
            {
                c_tensor[i16][j16]->zero();
            }
        }

        auto global_load_i = block_i.unfold() + global_load_idx.x().cast<I_Dim>();
        auto global_load_j = block_j.unfold() + global_load_idx.x().cast<J_Dim>();

        A_Loader loader_a(global_load_i < a.I);
        B_Loader loader_b(global_load_j < b.J);

        // Iterate over chunks of the k-dimension.
        for (auto iter : range_with_step<A_Fragments::K16.get()>(a.K16 + A_Fragments::K16))
        {
            if (iter < a.K16)
            {
                loader_a.load(smem_a_store.ping_pong(ping_pong).get(), a.get());
                loader_b.load(smem_b_store.ping_pong(ping_pong).get(), b.get());

                __pipeline_commit();
                a = a[A_Fragments::K16];
                b = b[B_Fragments::K16];
            }

            ping_pong ^= 1;

            if (iter > 0)
            {
                __pipeline_wait_prior(iter < Params::k16 ? 1 : 0);

                __syncthreads();

                A_Fragments::data_type a_data[A_Fragments::size];
                B_Fragments::data_type b_data[B_Fragments::size];

                A_Fragments a_tensor(a_data);
                B_Fragments b_tensor(b_data);

                for (auto k16 : a_tensor.K16)
                {
                    for (auto i16 : c_tensor.I16)
                    {
                        a_tensor[k16][i16]->load(smem_a_load.ping_pong(ping_pong)[k16][i16].get());
                    }
                    for (auto j16 : c_tensor.J16)
                    {
                        b_tensor[k16][j16]->load(smem_b_load.ping_pong(ping_pong)[k16][j16].get());
                    }
                }

                for (auto k16 : a_tensor.K16)
                {
                    for (auto i16 : c_tensor.I16)
                    {
                        for (auto j16 : c_tensor.J16)
                        {
                            mma_trans(
                                *c_tensor[i16][j16],
                                *a_tensor[k16][i16],
                                *b_tensor[k16][j16],
                                *c_tensor[i16][j16]);
                        }
                    }
                }

                __syncthreads();
            }
        }

        // Store outputs through shared memory.
        smem_a.deallocate(smem_allocator);
        smem_b.deallocate(smem_allocator);
        auto smem_c_array = smem_allocator.allocate<__half2>(SmemCStore::size);

        C_Fragments::data_type::Index c_idx(compute_idx.lane());
        auto smem_c_store = SmemCStore(smem_c_array)[compute_idx.i32()][compute_idx.j64()][c_idx.j2m4()];
        for (auto i16 : c_tensor.I16)
        {
            for (auto j16 : c_tensor.J16)
            {
                for (int f = 0; f < C_Fragments::data_type::size(); ++f)
                {
                    *smem_c_store[j16][c_idx.j8(f)][i16][c_idx.i(f)] = c_tensor[i16][j16]->to_half2(f);
                }
            }
        }

        // Transfer outputs from shared memory to global memory.
        auto c = C(c_ptr);
        auto smem_c_load = SmemCLoad(reinterpret_cast<const uint4 *>(smem_c_array))[compute_idx.i32()][compute_idx.j64()];
        for (int offset = compute_idx.lane().get(); offset < SmemCLoadIndex::size; offset += ComputeIndex::LANE.get())
        {
            SmemCLoadIndex idx(offset);
            auto i = block_i.unfold() + compute_idx.i32().unfold() + idx.i();
            auto j8 = block_j.fold<8>() + compute_idx.j64().fold<8>() + idx.j8();
            if (i < c.I && j8 < c.J8)
            {
                *c[i][j8] = *smem_c_load[idx.j8()][idx.i()];
            }
        }
    }
}