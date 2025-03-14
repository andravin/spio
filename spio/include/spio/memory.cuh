#ifndef SPIO_MEMORY_H_
#define SPIO_MEMORY_H_

#include <cuda_pipeline.h>

namespace spio
{
    /// @brief Convenience interface to CUDA's __pipline_memcpy_async.
    /// This template function simplifies the interface to the CUDA pipeline memcpy.
    /// It infers the load size from the data_type of the src and dst arguments. It
    /// also zero-fills the entire element if the mask is false.
    /// @param dst Destination pointer.
    /// @param src Source pointer.
    /// @param mask if false, this thread skips the memcpy and fills its element with zeros instead.
    template <typename data_type>
    __device__ void memcpy_async(data_type *dst, const data_type *src, bool mask = true)
    {
        __pipeline_memcpy_async(
            dst,
            src,
            sizeof(data_type),
            mask ? 0 : sizeof(data_type));
    }
}

#endif