#ifndef SPIO_ASYNC_STRIP_LOADER_H_
#define SPIO_ASYNC_STRIP_LOADER_H_

#include "spio/macros.h"
#include "spio/memory.cuh"

namespace spio {

    /// @brief Class that encapsulates the logic for loading a strip of data from global to
    /// shared memory asynchronously.
    /// @tparam SmemCursor The shared memory cursor type.
    /// @tparam GlobalCursor The global memory cursor type.
    /// @tparam smem_stride The stride in shared memory.
    /// @tparam global_stride The stride in global memory.
    /// @tparam num_loads The number of loads to perform.
    /// @tparam smem_buffer_stride The stride between shared memory buffers.
    /// @tparam global_buffer_stride The stride between global memory buffers.
    /// @details This class uses the CUDA async memcpy to load a 2D tile of
    /// data between global and shared memory. Each thread copies one vector
    /// of data per load, moving to the next position using the specified strides.
    ///
    /// The constructor accepts raw pointers, constructs cursors internally, and
    /// records inbounds status from the global cursor.
    template <typename SmemCursor, typename GlobalCursor, typename data_type, int smem_stride,
              int global_stride, int num_loads, int smem_buffer_stride, int global_buffer_stride,
              int num_buffers>
    class AsyncStripLoader {
        data_type* _smem_ptr;
        const data_type* _global_ptr;
        bool _mask;

    public:
        /// @brief Construct AsyncStripLoader from raw pointers, recording inbounds mask.
        /// @param smem Raw shared memory pointer at the starting position.
        /// @param global Raw global memory pointer at the starting position.
        __device__ AsyncStripLoader(data_type* smem, const data_type* global) {
            auto smem_cursor = SmemCursor(smem);
            auto global_cursor = GlobalCursor(global);
            _smem_ptr = smem_cursor.get();
            _global_ptr = global_cursor.get();
            _mask = global_cursor.inbounds();
        }

        __device__ void copy_async(int smem_buffer_idx, int global_buffer_idx) {
#pragma unroll

            for (int i = 0; i < num_loads; ++i) {
                memcpy_async(_smem_ptr + smem_buffer_idx * smem_buffer_stride + i * smem_stride,
                             _global_ptr + global_buffer_idx * global_buffer_stride +
                                 i * global_stride,
                             _mask);
            }
        }

        /// Prefetch the first buffer from global memory into the last shared memory buffer.
        /// The user must step the global pointer by 1 before calling copy_async().
        __device__ void prefetch_async() {
            constexpr int smem_buffer_idx = num_buffers - 1;
            constexpr int global_buffer_idx = 0;
            return copy_async(smem_buffer_idx, global_buffer_idx);
        }

        /// Copy the specified buffer from global memory into the specified shared memory buffer.
        /// The user must step the global buffer before calling copy_async again.
        __device__ void copy_async(int phase) {
            return copy_async(phase, phase);
        }

        /// @brief Advance the global memory pointer by a number of buffers.
        /// @param n Number of buffers to advance.
        __device__ void step(int n = num_buffers) {
            _global_ptr += n * global_buffer_stride;
        }
    };

    /// @brief 2D async strip loader for loading tiles with two iteration dimensions.
    /// @tparam SmemCursor The shared memory cursor type.
    /// @tparam GlobalCursor The global memory cursor type.
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
    /// The constructor accepts raw pointers, constructs cursors internally, and
    /// performs a dry-run iteration along the inner axis to record per-load inbounds masks.
    /// The outer dimension is assumed to be tested externally.
    template <typename SmemCursor, typename GlobalCursor, typename data_type, int smem_stride_inner,
              int global_stride_inner, int num_inner, int smem_stride_outer,
              int global_stride_outer, int num_outer, typename InnerStepDim, int inner_step_size,
              int smem_buffer_stride, int global_buffer_stride, int num_buffers>
    class AsyncStripLoader2D {
        data_type* _smem_ptr;
        const data_type* _global_ptr;
        bool _masks[num_inner];

    public:
        /// @brief Construct AsyncStripLoader2D from raw pointers, recording per-inner-load
        /// masks.
        /// @param smem Raw shared memory pointer at the starting position.
        /// @param global Raw global memory pointer at the starting position.
        __device__ AsyncStripLoader2D(data_type* smem, const data_type* global) {
            auto smem_cursor = SmemCursor(smem);
            auto global_cursor = GlobalCursor(global);
            _smem_ptr = smem_cursor.get();
            _global_ptr = global_cursor.get();
#pragma unroll
            for (int i = 0; i < num_inner; ++i) {
                _masks[i] = global_cursor[InnerStepDim(i * inner_step_size)].inbounds();
            }
        }

        __device__ void copy_async(int smem_buffer_idx, int global_buffer_idx) {
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

        __device__ void prefetch_async() {
            constexpr int smem_buffer_idx = num_buffers - 1;
            constexpr int global_buffer_idx = 0;
            return copy_async(smem_buffer_idx, global_buffer_idx);
        }

        __device__ void copy_async(int phase) {
            return copy_async(phase, phase);
        }

        /// @brief Advance the global memory pointer by a number of buffers.
        /// @param n Number of buffers to advance (default 1).
        __device__ void step(int n = num_buffers) {
            _global_ptr += n * global_buffer_stride;
        }
    };
}

#endif
