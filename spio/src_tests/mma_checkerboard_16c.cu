#include "spio.cuh"

// Include generated dimension and tensor classes.
#include "types.h"

extern "C" {
    using namespace spio;

    /// Test matrix multiplication with checkerboard layout.
    ///
    /// This kernel uses the checkerboard layout in shared memory
    /// when loading the A and B matrices.
    ///
    /// The checkerboard is a 16x2 grid with a vector
    /// of 8 half2 elements per cell. See checkerboard_index.h for details.
    ///
    /// - c_ptr: result matrix with float16 precision.
    /// - a_ptr: operand A matrix with float16 precision and format K16 x I x 16K
    /// - b_ptr: operand B matrix with float16 precision and format K16 x J x 16K
    __global__ void mma_checkerboard_16c(uint4* __restrict__ c_ptr, const uint4* __restrict__ a_ptr,
                                         const uint4* __restrict__ b_ptr) {
        // Allocate sufficient shared memory for the kernel.
        __shared__ uint4 smem[spio::max(ASmem::storage_size() + BSmem::storage_size(),
                                        CLoadSmem::storage_size())];

        // Allocate shared memory tensors for the A and B matrices.
        auto smem_allocator = StackAllocator(smem);
        auto a_smem = ASmem::allocate(smem_allocator);
        auto b_smem = BSmem::allocate(smem_allocator);

        // Get the tile coordinates for this thread-block.
        auto block_idx = make_coordinates(BLOCK_I(blockIdx.y), BLOCK_J(blockIdx.x));

        // Get the thread's coordinates for loads from global memory.
        auto global_load_idx = LoadGlobalIndex(threadIdx.x);

        // Cast coordinate dimensions for matrices A and B.
        auto a_global_load_idx = global_load_idx.cast<X, I>();
        auto b_global_load_idx = global_load_idx.cast<X, J>();

        // Construct cursors for loading A and B from global memory.
        auto a_global = AGlobal(a_ptr)[block_idx][a_global_load_idx].rebase();
        auto b_global = BGlobal(b_ptr)[block_idx][b_global_load_idx].rebase();

        // Construct cursors for storing A and B to shared memory.
        auto a_store_smem = ASmem(a_smem)[a_global_load_idx][XSwizzle(global_load_idx)].rebase();
        auto b_store_smem = BSmem(b_smem)[b_global_load_idx][XSwizzle(global_load_idx)].rebase();

        // Get the coordinates of the output tile this thread will compute.
        auto compute_idx = ComputeIndex(threadIdx.x);

        // Construct cursors for loading A and B from shared memory into tensor core fragments.
        auto a_reg_idx = AFragment::load_index_type(compute_idx);
        auto b_reg_idx = BFragment::load_index_type(compute_idx);
        auto a_load_smem = ASmem(a_smem)[compute_idx][ASwizzle(a_reg_idx)].rebase();
        auto b_load_smem = BSmem(b_smem)[compute_idx][BSwizzle(b_reg_idx)].rebase();

        // Initialize the accumulators.
        CReg::data_type c_data[CReg::storage_size()];
        auto c_reg = CReg(c_data);
        c_reg.zero();

        // Construct the global memory loaders for A and B. Set the valid mask.
        auto a_loader = ALoader(block_idx + a_global_load_idx < AGlobal::extents());
        auto b_loader = BLoader(block_idx + b_global_load_idx < BGlobal::extents());

        // Allocate registers for the A and B tiles.
        AReg::data_type a_data[AReg::storage_size()];
        BReg::data_type b_data[BReg::storage_size()];

        // Construct tensors for the A and B tiles.
        auto a_reg = AReg(a_data);
        auto b_reg = BReg(b_data);

        constexpr auto step = AReg::extent<K16>();
        constexpr auto size = AGlobal::extent<K>();

        // Prefetch the first k_chunk of data from A and B.
        if constexpr (size > K(0)) {
            a_loader.copy_async(a_store_smem.get(), a_global.get());
            b_loader.copy_async(b_store_smem.get(), b_global.get());
            __pipeline_commit();
            a_global.step(step);
            b_global.step(step);
        }

        // Main computation loop with pipelined memory operations.

        // Aggressive unrolling of the main loop improves arithmetic utilization.
#pragma unroll MAIN_LOOP_UNROLL_DEPTH
        for (auto k_double_chunk : range(K_DOUBLE_CHUNK(size))) {

            // Double-buffered loads and computation
            for (auto k_chunk : range(K_CHUNK(2))) {

                // If not the last iteration, copy the next tile from global
                // memory to shared memory asynchronously.
                if (k_double_chunk + k_chunk + K_CHUNK(1) < size) {
                    // Copy into the back-buffer.
                    a_loader.copy_async(a_store_smem[(k_chunk + 1) % 2].get(), a_global.get());
                    b_loader.copy_async(b_store_smem[(k_chunk + 1) % 2].get(), b_global.get());
                }

                // Advance the global memory tiles.
                a_global.step(step);
                b_global.step(step);

                // Synchronize on the previous iteration's global memory copy.
                __pipeline_commit();
                __pipeline_wait_prior(1);
                __syncthreads();

                // Load matrix tiles from shared memory into registers.
                a_reg.load(a_load_smem[k_chunk]);
                b_reg.load(b_load_smem[k_chunk]);

                // Matrix-multiply the tiles using Tensor Cores.
                // Compile-time type checking ensures the compatibility of the tile dimensions.
                mma(a_reg, b_reg, c_reg, c_reg);
                __syncthreads();
            }
        }
        __pipeline_wait_prior(0);

        // Final computation for any leftover iteration.
        if constexpr (K_DOUBLE_CHUNK(size) < size) {
            a_reg.load(a_load_smem);
            b_reg.load(b_load_smem);
            mma(a_reg, b_reg, c_reg, c_reg);
            __syncthreads();
        }

        // Store outputs to global memory via shared memory.
        a_smem.deallocate(smem_allocator);
        b_smem.deallocate(smem_allocator);
        auto c_smem = CStoreSmem::allocate(smem_allocator);

        // Transfer outputs from registers to shared memory, converting from float32 to float16.
        auto c_idx = CReg::data_type::compound_index_type(compute_idx);
        auto c_store_smem = c_smem[compute_idx][c_idx].rebase();
        for (auto e : range(c_reg)) {
            auto c_fragments = *c_reg[e];
            for (auto f : range(c_fragments)) {
                *c_store_smem[e][f] = __float22half2_rn(*c_fragments[f]);
            }
        }

        // Transfer outputs from shared memory to global memory.
        // Since each warp transfers its own transposed tile, no synchronization is needed.
        auto c_global = CGlobal(c_ptr)[block_idx][compute_idx].rebase();
        auto c_load_smem =
            CLoadSmem(reinterpret_cast<const uint4*>(c_smem.get()))[compute_idx].rebase();
        auto world_idx = block_idx + compute_idx;
        for (auto p : CLoadSmemIndex::partition<LANE>(compute_idx)) {
            if (world_idx + p < CGlobal::extents()) { *c_global[p] = *c_load_smem[p]; }
        }
    }
}