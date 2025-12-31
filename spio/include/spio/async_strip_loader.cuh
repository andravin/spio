#ifndef SPIO_ASYNC_STRIP_LOADER_H_
#define SPIO_ASYNC_STRIP_LOADER_H_

#include "spio/macros.h"
#include "spio/memory.cuh"

namespace spio {
    /// @brief Class that encapsulates the logic for loading a strip of data from global to shared
    /// memory asynchronously.
    /// @tparam smem_stride The stride in shared memory.
    /// @tparam global_stride The stride in global memory.
    /// @tparam num_loads The number of loads to perform.
    /// @details This class uses the CUDA async memcpy to load a 2D tile of
    /// data between global and shared memory. Each thread copies one vector
    /// of data per load, moving to the next position using the specified strides.
    ///
    /// Use StripLoaderParams to calculate the num_loads template parameter.
    ///
    /// Examples:
    ///
    /// 8 warps cooperate to load 16 units from an 8 x 2 tile using 2 loads:
    ///
    ///                       major_axis
    ///          +---+---+---+---+---+---+---+---+
    ///        0 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
    /// minor    +---+---+---+---+---+---+---+---+
    /// axis   1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
    ///          +---+---+---+---+---+---+---+---+
    ///        ^--load index       ^-- warp index
    ///
    /// 5 warps cooperate to load 10 units from a 5 x 2 tile using 2 loads:
    ///
    ///                  major_axis
    ///          +---+---+---+---+---+---+
    ///        0 | 0 | 1 | 2 | 3 | 4 | 5 |
    /// minor    +---+---+---+---+---+---+
    /// axis   1 | 0 | 1 | 2 | 3 | 4 | 5 |
    ///          +---+---+---+---+---+---+
    ///
    /// 8 warps cooperate to load 8 units from a 4 x 2 tile using 1 load:
    ///
    ///               major_axis
    ///          +---+---+---+---+
    ///        0 | 0 | 1 | 2 | 3 |
    /// minor    +---+---+---+---+
    /// axis   0 | 4 | 5 | 6 | 7 |
    ///          +---+---+---+---+
    ///
    template <int smem_stride, int global_stride, int num_loads> class AsyncStripLoader {
    public:
        /// @brief Construct AynscStripLoader and optionally mask-to-zero the current thread.
        /// @param mask Zero-fill the destination and skip the copy for the current thread with
        /// zeros if true.
        __device__ AsyncStripLoader(bool mask = true) : _mask(mask) {}

        /// @brief Copy data asynchronously from global to shared memory.
        /// @param smem_ptr Pointer to the shared memory destination for the current thread.
        /// @param global_ptr Pointer to the global memory source for the current thread.
        /// @tparam data_type The type of the data to load.
        template <typename data_type>
        __device__ void copy_async(data_type* smem_ptr, const data_type* global_ptr) {
            for (int i = 0; i < num_loads; ++i) {
                memcpy_async(smem_ptr + i * smem_stride, global_ptr + i * global_stride, _mask);
            }
        }

    private:
        bool _mask;
    };

    /// @brief 2D async strip loader for loading tiles with two iteration dimensions.
    /// @tparam smem_stride_inner Inner (major) axis stride in shared memory.
    /// @tparam global_stride_inner Inner (major) axis stride in global memory.
    /// @tparam num_inner Number of loads along inner (major) axis.
    /// @tparam smem_stride_outer Outer (minor) axis stride in shared memory.
    /// @tparam global_stride_outer Outer (minor) axis stride in global memory.
    /// @tparam num_outer Number of loads along outer (minor) axis.
    /// @details Extends AsyncStripLoader to handle 2D iteration patterns where each thread
    /// needs to load multiple elements along two dimensions (e.g., i/j and k16).
    ///
    /// Example: 4 warps loading a 128 x 2 tile (256 elements, 128 threads):
    ///   - Each thread loads 2 elements along i (inner) with stride 128
    ///   - Times 1 element along k16 (outer)
    ///   - Total: 2 loads per thread
    ///
    /// Example: 4 warps loading a 128 x 2 x 2 tile (512 elements, 128 threads):
    ///   - Each thread loads 2 elements along i (inner) with stride 128
    ///   - Times 2 elements along k16 (outer)
    ///   - Total: 4 loads per thread
    ///
    template <int smem_stride_inner, int global_stride_inner, int num_inner, int smem_stride_outer,
              int global_stride_outer, int num_outer>
    class AsyncStripLoader2D {
    public:
        __device__ AsyncStripLoader2D(bool mask = true) : _mask(mask) {}

        template <typename data_type>
        __device__ void copy_async(data_type* smem_ptr, const data_type* global_ptr) {
#pragma unroll
            for (int j = 0; j < num_outer; ++j) {
#pragma unroll
                for (int i = 0; i < num_inner; ++i) {
                    memcpy_async(smem_ptr + j * smem_stride_outer + i * smem_stride_inner,
                                 global_ptr + j * global_stride_outer + i * global_stride_inner,
                                 _mask);
                }
            }
        }

    private:
        bool _mask;
    };
}

#endif
