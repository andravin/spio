#include "spio.cuh"

// Include generated dimension and tensor classes.
// See tests/matmul/test_mma_checkerboard.py for details on the generation.
#include "types.h"

extern "C" {
    using namespace spio;

    /// Test matrix multiplication using typed tensors with dimensional projection.
    ///
    /// - c_ptr: result matrix with float16 precision.
    /// - a_ptr: operand A matrix with float16 precision and format K16 x I x 16K
    /// - b_ptr: operand B matrix with float16 precision and format K16 x J x 16K
    __global__ void mma_checkerboard_16c(uint4* __restrict__ c_ptr, const uint4* __restrict__ a_ptr,
                                         const uint4* __restrict__ b_ptr) {

        // Use dynamic shared memory.
        extern __shared__ char smem[];

        // Create an allocator for shared memory.
        auto smem_allocator = StackAllocator(smem);

        // Allocate shared memory for A and B.
        auto a_smem = ASmem::allocate(smem_allocator);
        auto b_smem = BSmem::allocate(smem_allocator);

        // Set up loaders for A and B.
        auto a_loader = ALoader(a_smem.get(), a_ptr);
        auto b_loader = BLoader(b_smem.get(), b_ptr);

        // Set up load views of the shared memory for A and B.
        auto a_load_smem = ALoadSmem(a_smem).rebase();
        auto b_load_smem = BLoadSmem(b_smem).rebase();

        // Allocate registers for the local matrices.
        AReg::data_type a_data[AReg::storage_size()];
        BReg::data_type b_data[BReg::storage_size()];
        CReg::data_type c_data[CReg::storage_size()];

        // Construct tensors for the local matrices.
        auto a_reg = AReg(a_data);
        auto b_reg = BReg(b_data);
        auto c_reg = CReg(c_data);
        c_reg.zero();

        // Get the size of the main loop.
        constexpr auto size = AGlobal::extent<K>();
        constexpr auto step = K_CHUNK(1).fold<16>();

        // Prefetch the first k_chunk of data from A and B.
        if constexpr (size > K(0)) {
            a_loader.prefetch_async();
            b_loader.prefetch_async();
            __pipeline_commit();
            a_loader.step(1);
            b_loader.step(1);
        }

#pragma unroll MAIN_LOOP_UNROLL_DEPTH
        for (auto k_double_chunk : range(K_DOUBLE_CHUNK(size))) {

            // Double-buffered loads and computation
#pragma unroll
            for (int phase = 0; phase < 2; ++phase) {
                // Synchronize on the previous iteration's global memory copy.
                __pipeline_wait_prior(0);
                __syncthreads();

                // If not the last iteration ..
                if (k_double_chunk + step * (phase + 1) < size) {
                    // .. copy the next tile into the back buffer.
                    a_loader.copy_async(phase);
                    b_loader.copy_async(phase);
                }
                __pipeline_commit();

                // Load matrix tiles from the front buffer.
                a_reg.load(a_load_smem[step * (1 - phase)]);
                b_reg.load(b_load_smem[step * (1 - phase)]);

                // Matrix multiply-accumulate the tiles using Tensor Cores.
                mma(a_reg, b_reg, c_reg, c_reg);
            }
            a_loader.step(2);
            b_loader.step(2);
        }

        // Final computation for any leftover iteration.
        if constexpr (K_DOUBLE_CHUNK(size) < size) {
            __pipeline_wait_prior(0);
            __syncthreads();
            a_reg.load(a_load_smem[step]);
            b_reg.load(b_load_smem[step]);
            mma(a_reg, b_reg, c_reg, c_reg);
        }

        // Separate smem loads from in the main loop from smem stores in the epilogue.
        __syncthreads();

        // Allocate shared memory for transposing the output matrix.
        a_smem.deallocate(smem_allocator);
        b_smem.deallocate(smem_allocator);
        auto c_smem = CSmem::allocate(smem_allocator);

        // Transfer outputs from registers to shared memory, converting from float32 to float16.
        auto c_store_smem = CStoreSmem(c_smem).rebase();
        for (auto e : range(c_reg)) {
            auto c_fragments = *c_reg[e];
            for (auto f : range(c_fragments)) {
                *c_store_smem[e][f] = __float22half2_rn(*c_fragments[f]);
            }
        }

        // Transfer outputs from shared memory to global memory.
        // Since each warp transfers its own transposed tile, no synchronization is needed.
        auto c_global = CGlobal(c_ptr);
        auto c_load_smem = CLoadSmem(c_smem).rebase();
        for (auto p : c_output_idx()) {
            if (c_global[p].inbounds()) { *c_global[p] = *c_load_smem[p]; }
        }
    }
}