#include "spio.cuh"

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
    /// of 8 half2 elements per cell. See checkerboard_index.h for details.
    ///
    /// @param c_ptr result matrix with float16 precision.
    /// @param a_ptr operand A matrix with float16 precision and format K16 x I X 16K
    /// @param b_ptr operand B matrix with float16 precision and format K16 x J X 16K
    __global__ void
    mma_checkerboard_16c(
        uint4 *__restrict__ c_ptr,
        const uint4 *__restrict__ a_ptr,
        const uint4 *__restrict__ b_ptr)
    {
        using J2M4 = Module<J, 4, 2>;

        __shared__ uint4 smem[spio::max(
            SmemA::storage_size() + SmemB::storage_size(),
            SmemCLoad::storage_size())];

        StackAllocator smem_allocator(smem);
        auto smem_a = SmemA::allocate(smem_allocator);
        auto smem_b = SmemB::allocate(smem_allocator);

        BLOCK_I block_i(blockIdx.y);
        BLOCK_J block_j(blockIdx.x);

        // Map the thread index to the global memory load index.
        GlobalLoadIndex global_load_idx(threadIdx.x);

        // Position the global memory input tiles.
        auto global_x16 = global_load_idx.get<X16>();
        auto global_x = global_x16.unfold() + global_load_idx.get<X>();
        auto a = A(a_ptr)[block_i.unfold() + global_x.cast<I>()][global_load_idx.get<K8>()].rebase();
        auto b = B(b_ptr)[block_j.unfold() + global_x.cast<J>()][global_load_idx.get<K8>()].rebase();

        // Define the mapping from the global memory tile to the shared memory tile.
        auto smem_checkers = Smem_Checkers(global_load_idx.get<X>(), global_load_idx.get<K8>()).get<CHECKERS>();
        auto smem_a_store = SmemA(smem_a)[global_x16.cast<I>()][smem_checkers].rebase();
        auto smem_b_store = SmemB(smem_b)[global_x16.cast<J>()][smem_checkers].rebase();

        // Define the mapping from the thread index to the warp's i32 x j64 tiles and the lane index.
        ComputeIndex compute_idx(threadIdx.x);

        // Define the mapping from the shared memory tile to the register tile.
        A_Tile::data_type::LoadIndex a_load_idx(compute_idx.get<LANE>().get());
        B_Tile::data_type::LoadIndex b_load_idx(compute_idx.get<LANE>().get());
        auto smem_a_checkers = SmemA_Checkers(a_load_idx.get<I>(), a_load_idx.get<K8>()).get<CHECKERS>();
        auto smem_b_checkers = SmemB_Checkers(b_load_idx.get<J>(), b_load_idx.get<K8>()).get<CHECKERS>();
        auto smem_a_load = SmemA(smem_a)[compute_idx.get<I32>().fold<16>()][smem_a_checkers].rebase();
        auto smem_b_load = SmemB(smem_b)[compute_idx.get<J64>().fold<16>()][smem_b_checkers].rebase();

        // Initialize the accumulator.
        C_Tile::data_type c_data[C_Tile::storage_size()];
        C_Tile c_tile(c_data);
        c_tile.zero();

        auto global_load_i = block_i.unfold() + global_load_idx.get<X>().cast<I>();
        auto global_load_j = block_j.unfold() + global_load_idx.get<X>().cast<J>();

        A_Loader loader_a(global_load_i < A::size<I>());
        B_Loader loader_b(global_load_j < B::size<J>());

        A_Tile::data_type a_data[A_Tile::storage_size()];
        B_Tile::data_type b_data[B_Tile::storage_size()];

        A_Tile a_tile(a_data);
        B_Tile b_tile(b_data);

        constexpr auto size = A::size<K16>();
        constexpr auto step_size = A_Tile::size<K16>();

        loader_a.load(smem_a_store.get(), a.get());
        loader_b.load(smem_b_store.get(), b.get());
        __pipeline_commit();
        a.step(step_size);
        b.step(step_size);

        // Compute.
        for (int iter = 0; iter < size.get(); iter += 2 * step_size.get())
        {
            for (auto phase : range(PING(2)))
            {
                if (iter + 1 + phase.get() < size.get() - step_size.get())
                {
                    loader_a.load(smem_a_store[(phase + 1) % 2].get(), a.get());
                    loader_b.load(smem_b_store[(phase + 1) % 2].get(), b.get());
                }
                a.step(step_size);
                b.step(step_size);
                __pipeline_commit();
                __pipeline_wait_prior(1);
                __syncthreads();
                a_tile.load(smem_a_load[phase]);
                b_tile.load(smem_b_load[phase]);
                mma(a_tile, b_tile, c_tile, c_tile);
                __syncthreads();
            }
        }
        __pipeline_wait_prior(0);

        // Final compute for any leftover step.
        int last_step = size.get() - size.get() % step_size.get();
        if (last_step < size.get())
        {
            a_tile.load(smem_a_load);
            b_tile.load(smem_b_load);
            mma(a_tile, b_tile, c_tile, c_tile);
            __syncthreads();
        }

        // Store outputs through shared memory.
        C_Tile::data_type::Index c_idx(compute_idx.get<LANE>().get());

        smem_a.deallocate(smem_allocator);
        smem_b.deallocate(smem_allocator);

        auto smem_c = SmemCStore::allocate(smem_allocator);
        auto smem_c_store = smem_c[compute_idx.get<I32>()]
                                  [compute_idx.get<J64>().fold<8>()]
                                  [c_idx.get<J2M4>().cast<J2>()]
                                      .rebase();

        for (int f = 0; f < C_Tile::data_type::size(); ++f)
        {
            auto smem_c_cursor = smem_c_store[c_idx.get<J8>(f)][c_idx.get<I>(f)].rebase();
            for (auto i16 : range(c_tile.size<I16>()))
            {
                for (auto j16 : range(c_tile.size<J16>()))
                {
                    *smem_c_cursor[j16.fold<8>()][i16] = c_tile[i16][j16]->to_half2(f);
                }
            }
        }

        // Transfer outputs from shared memory to global memory.
        auto c = C(c_ptr);
        auto smem_c_load_tensor = SmemCLoad(reinterpret_cast<const uint4 *>(smem_c.get()));
        auto smem_c_load = smem_c_load_tensor[compute_idx.get<I32>()][compute_idx.get<J64>().fold<8>()].rebase();
        for (int offset = compute_idx.get<LANE>().get(); offset < SmemCLoadIndex::size(); offset += ComputeIndex::size<LANE>().get())
        {
            SmemCLoadIndex idx(offset);
            auto i = block_i.unfold() + compute_idx.get<I32>().unfold() + idx.get<I>();
            auto j8 = block_j.fold<8>() + compute_idx.get<J64>().fold<8>() + idx.get<J8>();
            if (i < c.size<I>() && j8 < c.size<J8>())
            {
                *c[i][j8] = *smem_c_load[idx.get<J8>()][idx.get<I>()];
            }
        }
    }
}