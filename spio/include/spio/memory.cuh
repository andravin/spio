#ifndef SPIO_MEMORY_H_
#define SPIO_MEMORY_H_

#include <cuda_pipeline.h>

namespace spio {
    /// @brief Convenience interface to CUDA's __pipline_memcpy_async.
    /// This template function simplifies the interface to the CUDA pipeline memcpy.
    /// It infers the load size from the data_type of the src and dst arguments. It
    /// also zero-fills the entire element if the mask is false.
    /// @param dst Destination pointer.
    /// @param src Source pointer.
    /// @param mask if false, this thread skips the memcpy and fills its element with zeros instead.
    ///
    /// Uses inline PTX instead of __pipeline_memcpy_async() because the intrinsic
    /// generates redundant code: paired LDGSTS instructions with duplicated address
    /// arithmetic for the in-bounds and out-of-bounds paths, even though only one
    /// path executes. The inline PTX generates a single LDGSTS with the src-size
    /// operand controlling zero-fill behavior.
    template <typename data_type>
    __device__ void memcpy_async(data_type* dst, const data_type* __restrict__ src,
                                 bool mask = true) {
        constexpr auto size = static_cast<uint32_t>(sizeof(data_type));
        const auto smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem), "l"(src), "r"(mask ? size : 0));
    }
}

#endif