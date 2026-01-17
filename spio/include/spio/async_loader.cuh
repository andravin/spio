#ifndef SPIO_ASYNC_LOADER_H_
#define SPIO_ASYNC_LOADER_H_

#include "spio/macros.h"
#include "spio/memory.cuh"

namespace spio {

    /// Encapsulates async loading of data strips from global to shared memory.
    ///
    /// Uses CUDA async memcpy to load 2D tiles. Each thread copies one vector per load,
    /// moving to the next position using the specified strides. Records inbounds status
    /// from the global cursor.
    template <typename SmemCursor, typename GlobalCursor, typename data_type, int smem_stride,
              int global_stride, int num_loads, int smem_buffer_stride, int global_buffer_stride,
              int num_buffers>
    class AsyncLoader {
        data_type* _smem_ptr;
        const data_type* _global_ptr;
        bool _mask;

    public:
        /// Constructs AsyncLoader from raw pointers, recording inbounds mask.
        ///
        /// Parameters:
        ///   smem     Raw shared memory pointer at the starting position.
        ///   global   Raw global memory pointer at the starting position.
        __device__ AsyncLoader(data_type* smem, const data_type* global) {
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

        /// Advances the global memory pointer by a number of buffers.
        ///
        /// Parameters:
        ///   n   Number of buffers to advance.
        __device__ void step(int n = num_buffers) {
            _global_ptr += n * global_buffer_stride;
        }
    };

    /// 2D async strip loader for tiles with two iteration dimensions.
    ///
    /// Extends AsyncLoader for 2D iteration patterns where threads load multiple elements
    /// along two dimensions. Records per-load inbounds masks via dry-run iteration.
    template <typename SmemCursor, typename GlobalCursor, typename data_type, int smem_stride_inner,
              int global_stride_inner, int num_inner, int smem_stride_outer,
              int global_stride_outer, int num_outer, typename InnerStepDim, int inner_step_size,
              int smem_buffer_stride, int global_buffer_stride, int num_buffers>
    class AsyncLoader2D {
        data_type* _smem_ptr;
        const data_type* _global_ptr;
        bool _masks[num_inner];

    public:
        /// Constructs AsyncLoader2D from raw pointers, recording per-inner-load masks.
        ///
        /// Parameters:
        ///   smem     Raw shared memory pointer at the starting position.
        ///   global   Raw global memory pointer at the starting position.
        __device__ AsyncLoader2D(data_type* smem, const data_type* global) {
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

        /// Advances the global memory pointer by a number of buffers.
        ///
        /// Parameters:
        ///   n   Number of buffers to advance.
        __device__ void step(int n = num_buffers) {
            _global_ptr += n * global_buffer_stride;
        }
    };
}

#endif
