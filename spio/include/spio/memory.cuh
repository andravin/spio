#ifndef SPIO_MEMORY_H_
#define SPIO_MEMORY_H_

#include <cuda_pipeline.h>

namespace spio {
    /// Convenience interface to CUDA's async memcpy with zero-fill support.
    ///
    /// Simplifies the CUDA pipeline memcpy interface. Infers load size from data_type.
    /// Uses inline PTX for optimal code generation.
    ///
    /// Parameters:
    ///   dst    Destination pointer in shared memory.
    ///   src    Source pointer in global memory.
    ///   mask   If false, zero-fills instead of copying.
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