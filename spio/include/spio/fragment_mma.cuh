#ifndef SPIO_FRAGMENT_MMA_CUH
#define SPIO_FRAGMENT_MMA_CUH

#include "fragment.cuh"
#include "mma.cuh"
#include "dim.h"

namespace spio
{

    /// Performs D = A x trans(B) + C matrix-multiplication with tensor cores.
    template <typename I, typename J, typename K>
    __device__ void mma_trans(MMA_M16_N8_F32_C<I, J> &d, const MMA_M16_K8_F16_A<I, K> &a, const MMA_N8_K8_F16_B<K, J> &b, const MMA_M16_N8_F32_C<I, J> &c)
    {
        mma_m16_n8_k8(d.vec4(), a.reg2(), b.reg(), c.vec4());
    }

    /// Performs D = A x trans(B) + C matrix-multiplication with tensor cores.
    template <typename I, typename J, typename K>
    __device__ void mma_trans(MMA_M16_N8_F32_C<I, J> &d, const MMA_M16_K16_F16_A<I, K> &a, const MMA_N8_K16_F16_B<K, J> &b, const MMA_M16_N8_F32_C<I, J> &c)
    {
        mma_m16_n8_k16(d.vec4(), a.reg4(), b.reg2(), c.vec4());
    }

    /// Performs D = A x trans(B) + C matrix multiply-accumulate with tensor cores.
    template <typename I, typename J, typename K>
    __device__ void mma_trans(MMA_M16_N16_F32_C<I, J> &d, const MMA_M16_K16_F16_A<I, K> &a, const MMA_N16_K16_F16_B<K, J> &b, const MMA_M16_N16_F32_C<I, J> &c)
    {
        mma_m16_n8_k16(d.vec4(0), a.reg4(), b.reg2(0), c.vec4(0));
        mma_m16_n8_k16(d.vec4(1), a.reg4(), b.reg2(1), c.vec4(1));
    }

    /// Performs D = A x trans(B) + C with tensor cores, B fragments in reverse order.
    template <typename I, typename J, typename K>
    __device__ void mma_trans_reverse(MMA_M16_N16_F32_C<I, J> &d, const MMA_M16_K16_F16_A<I, K> &a, const MMA_N16_K16_F16_B<K, J> &b, const MMA_M16_N16_F32_C<I, J> &c)
    {
        // Process B fragments in reverse order
        mma_m16_n8_k16(d.vec4(1), a.reg4(), b.reg2(1), c.vec4(1));
        mma_m16_n8_k16(d.vec4(0), a.reg4(), b.reg2(0), c.vec4(0));
    }

    /// Performs D = A x trans(B) + C with tensor cores (reverse, identical for 8x8 B).
    template <typename I, typename J, typename K>
    __device__ void mma_trans_reverse(MMA_M16_N8_F32_C<I, J> &d, const MMA_M16_K8_F16_A<I, K> &a, const MMA_N8_K8_F16_B<K, J> &b, const MMA_M16_N8_F32_C<I, J> &c)
    {
        // For single B fragments, just delegate to the regular version
        mma_trans(d, a, b, c);
    }

    /// Performs D = A x trans(B) + C with tensor cores (reverse, identical for 8x16 B).
    template <typename I, typename J, typename K>
    __device__ void mma_trans_reverse(MMA_M16_N8_F32_C<I, J> &d, const MMA_M16_K16_F16_A<I, K> &a, const MMA_N8_K16_F16_B<K, J> &b, const MMA_M16_N8_F32_C<I, J> &c)
    {
        // For single B fragments, just delegate to the regular version
        mma_trans(d, a, b, c);
    }

}

#endif // SPIO_FRAGMENT_MMA_CUH
