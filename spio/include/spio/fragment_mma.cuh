#ifndef SPIO_FRAGMENT_MMA_CUH
#define SPIO_FRAGMENT_MMA_CUH

#include "fragment.cuh"
#include "mma.cuh"
#include "dim.h"

namespace spio
{

    /// @brief  Perform D = A x trans(B) + C matrix-multiplication with tensor cores.
    /// @param d Result matrix D is a 16 x 8 fragment in float32.
    /// @param a Matrix argument A is a 16 x 8 fragment in float16.
    /// @param b Matrix argument B is an 8 x 8 fragment in float16.
    /// @param c Accumulate matrix C is a 16 x 8 fragment in float32.
    template <typename I, typename J, typename K>
    __device__ void mma_trans(MMA_M16_N8_F32_C<I, J> &d, const MMA_M16_K8_F16_A<I, K> &a, const MMA_N8_K8_F16_B<K, J> &b, const MMA_M16_N8_F32_C<I, J> &c)
    {
        mma_m16_n8_k8(d.vec4(), a.reg2(), b.reg(), c.vec4());
    }

    /// @brief Perform D = A x trans(B) + C matrix-multiplication with tensor cores.
    /// @param d Result matrix D is a 16 x 8 fragment in float32.
    /// @param a Matrix argument A is a 16 x 16 fragment in float16.
    /// @param b Matrix argument B is an 8 x 16 fragment in float16.
    /// @param c Accumulate matrix C is a 16 x 8 fragment in float32.
    template <typename I, typename J, typename K>
    __device__ void mma_trans(MMA_M16_N8_F32_C<I, J> &d, const MMA_M16_K16_F16_A<I, K> &a, const MMA_N8_K16_F16_B<K, J> &b, const MMA_M16_N8_F32_C<I, J> &c)
    {
        mma_m16_n8_k16(d.vec4(), a.reg4(), b.reg2(), c.vec4());
    }

    /// @brief Perform D = A x trans(B) + C matrix multiply-accumulate with tensor cores.
    /// @param d Result matrix D is a 16 x 16 fragment in float32.
    /// @param a Matrix argument A is a 16 x 16 fragment in float16.
    /// @param b Matrix argument B is an 16 x 16 fragment in float16.
    /// @param c Accumulate matrix C is a 16 x 16 fragment in float32.
    template <typename I, typename J, typename K>
    __device__ void mma_trans(MMA_M16_N16_F32_C<I, J> &d, const MMA_M16_K16_F16_A<I, K> &a, const MMA_N16_K16_F16_B<K, J> &b, const MMA_M16_N16_F32_C<I, J> &c)
    {
        mma_m16_n8_k16(d.vec4(0), a.reg4(), b.reg2(0), c.vec4(0));
        mma_m16_n8_k16(d.vec4(1), a.reg4(), b.reg2(1), c.vec4(1));
    }

    template <typename I, typename J, typename K, typename D, typename A, typename B, typename C>
    __device__ void tensor_mma_trans_ijk(D &d, const A &a, const B &b, const C &c)
    {
        for (auto k : range(a.template size<K>()))
        {
            for (auto i : range(c.template size<I>()))
            {
                for (auto j : range(c.template size<J>()))
                {
                    mma_trans(*d[i][j], *a[k][i], *b[k][j], *c[i][j]);
                }
            }
        }
    }

    template <typename I, typename J, typename D, typename A, typename B, typename C>
    __device__ void tensor_mma_trans_ij(D &d, const A &a, const B &b, const C &c)
    {
        for (auto i : range(c.template size<I>()))
        {
            for (auto j : range(c.template size<J>()))
            {
                mma_trans(*d[i][j], *a[i], *b[j], *c[i][j]);
            }
        }
    }

    template <typename I, typename D, typename A, typename B, typename C>
    __device__ void tensor_mma_trans_i(D &d, const A &a, const B &b, const C &c)
    {
        for (auto i : range(c.template size<I>()))
        {
            mma_trans(*d[i], *a[i], *b, *c[i]);
        }
    }

    template <typename D, typename A, typename B, typename C>
    __device__ void tensor_mma_trans(D &d, const A &a, const B &b, const C &c)
    {
        mma_trans(*d, *a, *b, *c);
    }

}

#endif // SPIO_FRAGMENT_MMA_CUH