#ifndef SPIO_ASYNC_STRIP_LOADER_H_
#define SPIO_ASYNC_STRIP_LOADER_H_

#include <cuda_pipeline_primitives.h>

#include "spio/macros.h"

namespace spio
{
    /// @brief Class that encapsulates the logic for loading a strip of data from global to shared memory asynchronously.
    /// @tparam smem_stride The stride in shared memory.
    /// @tparam global_stride The stride in global memory.
    /// @tparam num_loads The number of loads to perform.
    /// @tparam veclen The number of bytes per load in each thread.
    /// @details This class uses the CUDA async memcpy to load a 2D tile of
    /// data from global memory tensor to a shared memory tensor.
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
    template <int smem_stride, int global_stride, int num_loads, int veclen = 16>
    class AsyncStripLoader
    {
        __device__ constexpr static int _compute_zfill(bool mask) { return mask ? 0 : veclen; }

    public:
        /// @brief Construct AynscStripLoader and optionally mask-to-zero the current thread.
        /// @param mask Zero-fill the destination and skip the copy for the current thread with zeros if true.
        __device__ AsyncStripLoader(bool mask = true)
            : _zfill(_compute_zfill(mask))
        {
        }

        /// @param smem_ptr Pointer to the shared memory destination for the current thread.
        /// @param global_ptr Pointer to the global memory source for the current thread.
        __device__ void load(uint4 *smem_ptr, const uint4 *global_ptr)
        {
            for (int i = 0; i < num_loads; ++i)
            {
                __pipeline_memcpy_async(smem_ptr + i * smem_stride, global_ptr + i * global_stride, veclen, _zfill);
            }
        }

    private:
        int _zfill;
    };
}

#endif
