#ifndef SPIO_ASYNC_STRIP_LOADER_H_
#define SPIO_ASYNC_STRIP_LOADER_H_

#include "spio/macros.h"
#include "spio/memory.cuh"

namespace spio {
    /// @brief Class that encapsulates the logic for loading a strip of data from global to shared
    /// memory asynchronously.
    /// @tparam data_type The data type to load.
    /// @tparam smem_stride The stride in shared memory.
    /// @tparam global_stride The stride in global memory.
    /// @tparam num_loads The number of loads to perform.
    /// @tparam smem_buffer_stride The stride between shared memory buffers.
    /// @tparam global_buffer_stride The stride between global memory buffers.
    /// @details This class uses the CUDA async memcpy to load a 2D tile of
    /// data between global and shared memory. Each thread copies one vector
    /// of data per load, moving to the next position using the specified strides.
    ///
    /// The constructor accepts a global memory cursor and records inbounds status.
    template <typename data_type, int smem_stride, int global_stride, int num_loads,
              int smem_buffer_stride, int global_buffer_stride>
    class AsyncStripLoader {
        const data_type* _global_ptr;
        data_type* _smem_ptr;
        bool _mask;

    public:
        /// @brief Construct AsyncStripLoader from a cursor, recording inbounds mask.
        /// @param smem Shared memory cursor at the starting position.
        /// @param global Global memory cursor at the starting position.
        template <typename SmemCursor, typename GlobalCursor>
        __device__ AsyncStripLoader(SmemCursor smem, GlobalCursor global)
            : _global_ptr(global.get()),
              _smem_ptr(smem.get()),
              _mask(global.inbounds()) {}

        /// @brief Copy data asynchronously from global to shared memory.
        __device__ void copy_async(int smem_buffer_idx = 0, int global_buffer_idx = 0) {
#pragma unroll
            for (int i = 0; i < num_loads; ++i) {
                memcpy_async(_smem_ptr + smem_buffer_idx * smem_buffer_stride + i * smem_stride,
                             _global_ptr + global_buffer_idx * global_buffer_stride +
                                 i * global_stride,
                             _mask);
            }
        }

        __device__ void step(int num_buffers = 1) {
            _global_ptr += num_buffers * global_buffer_stride;
        }
    };

    /// @brief 2D async strip loader for loading tiles with two iteration dimensions.
    /// @tparam data_type The data type to load.
    /// @tparam smem_stride_inner Inner (major) axis stride in shared memory.
    /// @tparam global_stride_inner Inner (major) axis stride in global memory.
    /// @tparam num_inner Number of loads along inner (major) axis.
    /// @tparam smem_stride_outer Outer (minor) axis stride in shared memory.
    /// @tparam global_stride_outer Outer (minor) axis stride in global memory.
    /// @tparam num_outer Number of loads along outer (minor) axis.
    /// @tparam InnerStepDim The dimension type to step by along the inner axis.
    /// @tparam inner_step_size The step size in InnerStepDim units.
    /// @tparam smem_buffer_stride The stride between shared memory buffers.
    /// @tparam global_buffer_stride The stride between global memory buffers.
    /// @details Extends AsyncStripLoader to handle 2D iteration patterns where each thread
    /// needs to load multiple elements along two dimensions (e.g., i/j and k16).
    ///
    /// The constructor accepts a global memory cursor and performs a dry-run
    /// iteration along the inner axis to record per-load inbounds masks.
    /// The outer dimension is assumed to be tested externally.
    template <typename data_type, int smem_stride_inner, int global_stride_inner, int num_inner,
              int smem_stride_outer, int global_stride_outer, int num_outer, typename InnerStepDim,
              int inner_step_size, int smem_buffer_stride, int global_buffer_stride>
    class AsyncStripLoader2D {
        data_type* _smem_ptr;
        const data_type* _global_ptr;
        bool _masks[num_inner];

    public:
        /// @brief Construct AsyncStripLoader2D from a cursor, recording per-inner-load masks.
        /// @param smem Shared memory cursor at the starting position.
        /// @param global Global memory cursor at the starting position.
        template <typename SmemCursor, typename GlobalCursor>
        __device__ AsyncStripLoader2D(SmemCursor smem, GlobalCursor global)
            : _smem_ptr(smem.get()),
              _global_ptr(global.get()) {
#pragma unroll
            for (int i = 0; i < num_inner; ++i) {
                _masks[i] = global[InnerStepDim(i * inner_step_size)].inbounds();
            }
        }

        /// @brief Copy data asynchronously from global to shared memory.
        /// @param smem_buffer_idx Index of the shared memory buffer to load into.
        /// @param global_buffer_idx Index of the global memory buffer to load from.
        __device__ void copy_async(int smem_buffer_idx = 0, int global_buffer_idx = 0) {
#pragma unroll
            for (int i = 0; i < num_inner; ++i) {
#pragma unroll
                for (int j = 0; j < num_outer; ++j) {
                    memcpy_async(_smem_ptr + smem_buffer_idx * smem_buffer_stride +
                                     i * smem_stride_inner + j * smem_stride_outer,
                                 _global_ptr + global_buffer_idx * global_buffer_stride +
                                     i * global_stride_inner + j * global_stride_outer,
                                 _masks[i]);
                }
            }
        }

        /// @brief Advance the global memory pointer by a number of buffers.
        __device__ void step(int num_buffers = 1) {
            _global_ptr += num_buffers * global_buffer_stride;
        }
    };
}

#endif
