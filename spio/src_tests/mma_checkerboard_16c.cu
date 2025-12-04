#include "spio.cuh"

// Include generated dimension and tensor classes.
#include "types.h"

extern "C" {
    using namespace spio;

    /// Test matrix multiplication with checkerboard layout.
    ///
    /// This kernel uses the checkerboard layout for shared memory when
    /// loading the A and B matrices.
    ///
    /// The checkerboard is a 16x2 grid with a vector
    /// of 8 half2 elements per cell. See checkerboard_index.h for details.
    ///
    /// - c_ptr: result matrix with float16 precision.
    /// - a_ptr: operand A matrix with float16 precision and format K16 x I X 16K
    /// - b_ptr: operand B matrix with float16 precision and format K16 x J X 16K
    __global__ void mma_checkerboard_16c(uint4* __restrict__ c_ptr, const uint4* __restrict__ a_ptr,
                                         const uint4* __restrict__ b_ptr) {
        // Allocate sufficient shared memory for the kernel.
        __shared__ uint4 smem[spio::max(SmemA::storage_size() + SmemB::storage_size(),
                                        SmemCLoad::storage_size())];

        // Allocate shared memory tensors for double-buffering loads from matrices A and B.
        StackAllocator smem_allocator(smem);
        auto smem_a = SmemA::allocate(smem_allocator);
        auto smem_b = SmemB::allocate(smem_allocator);

        // Get the tile coordinates for this thread-block.
        auto block_idx = make_coordinates(BLOCK_I(blockIdx.y), BLOCK_J(blockIdx.x));

        // Map this thread to the global memory load index.
        GlobalLoadIndex global_load_idx(threadIdx.x);

        // Map global load index into A/B tensor coordinates. Replace base dims X with I/J.
        auto global_load_a_idx = global_load_idx.cast<X, I>();
        auto global_load_b_idx = global_load_idx.cast<X, J>();
        auto a = A(a_ptr)[block_idx][global_load_a_idx].rebase();
        auto b = B(b_ptr)[block_idx][global_load_b_idx].rebase();

        // Map the global memory tile to the shared memory tile.
        auto smem_checkers = Smem_Checkers(global_load_idx);
        auto smem_a_store = SmemA(smem_a)[global_load_a_idx][smem_checkers].rebase();
        auto smem_b_store = SmemB(smem_b)[global_load_b_idx][smem_checkers].rebase();

        // Get the tile coordinates for this thread.
        ComputeIndex compute_idx(threadIdx.x);

        // Map the shared memory tile to the registers.
        A_Tile::data_type::LoadIndex a_load_idx(compute_idx.get<LANE>().get());
        B_Tile::data_type::LoadIndex b_load_idx(compute_idx.get<LANE>().get());
        auto smem_a_checkers = SmemA_Checkers(a_load_idx);
        auto smem_b_checkers = SmemB_Checkers(b_load_idx);
        auto smem_a_load = SmemA(smem_a)[compute_idx][smem_a_checkers].rebase();
        auto smem_b_load = SmemB(smem_b)[compute_idx][smem_b_checkers].rebase();

        // Initialize the accumulators.
        C_Tile::data_type c_data[C_Tile::storage_size()];
        C_Tile c_tile(c_data);
        c_tile.zero();

        // Construct the global memory loaders for A and B.
        // Set the valid mask based on the tile coordinates.
        A_Loader loader_a(block_idx + global_load_a_idx < A::sizes());
        B_Loader loader_b(block_idx + global_load_b_idx < B::sizes());

        // Allocate registers for the A and B tiles.
        A_Tile::data_type a_data[A_Tile::storage_size()];
        B_Tile::data_type b_data[B_Tile::storage_size()];

        // Construct tensors for the A and B tiles.
        A_Tile a_tile(a_data);
        B_Tile b_tile(b_data);

        constexpr auto size = A::size<K16>();
        constexpr auto step_size = A_Tile::size<K16>();

        // Prefetch the first chunk of data from A and B.
        if constexpr (size > 0) {
            loader_a.copy_async(smem_a_store.get(), a.get());
            loader_b.copy_async(smem_b_store.get(), b.get());
            __pipeline_commit();
            a.step(step_size);
            b.step(step_size);
        }

        // Main computation loop with pipelined memory operations.

        // Aggressive unrolling of the main loop improves arithmetic utilization.
#pragma unroll MAIN_LOOP_UNROLL_DEPTH
        for (int iter = 0; iter < size.get(); iter += 2 * step_size.get()) {

            // Double-buffer loads and compute.
            for (auto phase : range(PING(2))) {

                // If not the last iteration, copy the next tile from global
                // memory to shared memory asynchronously.
                if (iter + (phase.get() + 1) * step_size.get() < size.get()) {
                    // Copy into the back-buffer.
                    loader_a.copy_async(smem_a_store[(phase + 1) % 2].get(), a.get());
                    loader_b.copy_async(smem_b_store[(phase + 1) % 2].get(), b.get());
                }

                // Advance the global memory tiles.
                a.step(step_size);
                b.step(step_size);

                // Synchronize on the previous iteration's global memory copy.
                __pipeline_commit();
                __pipeline_wait_prior(1);
                __syncthreads();

                // Load matrix tiles from shared memory into registers.
                a_tile.load(smem_a_load[phase]);
                b_tile.load(smem_b_load[phase]);

                // Matrix-multiply the tiles using Tensor Cores.
                // Compile-time type checking ensures the compatibility of the tile dimensions.
                mma(a_tile, b_tile, c_tile, c_tile);
                __syncthreads();
            }
        }
        __pipeline_wait_prior(0);

        // Final compute for any leftover step.
        if constexpr (size % (step_size * 2) != 0) {
            a_tile.load(smem_a_load);
            b_tile.load(smem_b_load);
            mma(a_tile, b_tile, c_tile, c_tile);
            __syncthreads();
        }

        // Store outputs through shared memory.
        smem_a.deallocate(smem_allocator);
        smem_b.deallocate(smem_allocator);
        auto smem_c = SmemCStore::allocate(smem_allocator);

        // Transfer outputs from registers to shared memory, converting from fp32 to fp16.
        C_Tile::data_type::CompoundIndex c_idx(compute_idx.get<LANE>().get());
        auto smem_c_base = smem_c[compute_idx][c_idx.base_coord()].rebase();
        for (int f = 0; f < C_Tile::data_type::size(); ++f) {
            auto smem_c_fragment = smem_c_base[c_idx.fragment_coord(f)].rebase();
            for (auto coord : range(c_tile)) {
                *smem_c_fragment[coord] = c_tile[coord]->to_half2(f);
            }
        }

        // Transfer outputs from shared memory to global memory.
        // Each warp transfers its own transposed tile, so no synchronization is needed.
        auto c = C(c_ptr)[block_idx][compute_idx].rebase();
        auto smem_c_load_tensor = SmemCLoad(reinterpret_cast<const uint4*>(smem_c.get()));
        auto smem_c_load = smem_c_load_tensor[compute_idx].rebase();
        auto base_idx = block_idx + compute_idx;
        for (int offset = compute_idx.get<LANE>().get(); offset < SmemCLoadIndex::size();
             offset += ComputeIndex::size<LANE>().get()) {
            SmemCLoadIndex idx(offset);
            if (base_idx + idx < C::sizes()) { *c[idx] = *smem_c_load[idx]; }
        }
    }
}