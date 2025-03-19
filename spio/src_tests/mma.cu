#include <cuda_pipeline_primitives.h>

#include "spio/fragment_mma.cuh"
#include "spio/ldmatrix.cuh"
#include "spio/fragment_load_index.h"
#include "spio/fragment_index.h"
#include "spio/checkerboard_index.h"

extern "C"
{
    using namespace spio;

    /// @brief  Test mma.m16n8k8 with float16 data.
    /// @param A 16m x 8k matrix with float16 elements.
    /// @param B_trans 8n x 8k matrix with float16 elements.
    /// @param c_ptr 16m x 8n matrix with float32 elements.
    __global__ void mma_m16_n8_k8(
        float2 *__restrict__ c_ptr,
        const __half2 *__restrict__ a_ptr,
        const __half2 *__restrict__ b_trans_ptr)

    {
        class I : public Dim<I> { using Dim::Dim; };
        class J : public Dim<J> { using Dim::Dim; };
        class K : public Dim<K> { using Dim::Dim; };

        using A = MMA_M16_K8_F16_A<I, K>;
        using B = MMA_N8_K8_F16_B<K, J>;
        using C = MMA_M16_N8_F32_C<I, J>;

        int lane = threadIdx.x % 32;

        A a;
        A::Index a_index(lane);

        for (int f = 0; f < a.size(); ++f) {
            a(f) = a_ptr[a_index.get<I>(f).get() * 4 + a_index.get<Fold<K, 2>>(f).get()];
        }

        B b;
        B::Index b_index(lane);
        b() = b_trans_ptr[b_index.get<J>().get() * 4 + b_index.get<Fold<K, 2>>().get()];

        C c;
        c.zero();

        mma_trans(c, a, b, c);

        C::Index idx(lane);       
        c_ptr[idx.get<I>(0).get() * 4 + idx.get<Fold<J, 2>>(0).get()] = c.fragment(0);
        c_ptr[idx.get<I>(1).get() * 4 + idx.get<Fold<J, 2>>(1).get()] = c.fragment(1);
    }

    /// @brief  Test mma.m16n8k16 with float16 data.
    /// @param A 16m x 16k matrix with float16 elements.
    /// @param B_trans 8n x 16k matrix with float16 elements.
    /// @param C 16m x 8n matrix with float32 elements.
    __global__ void mma_m16_n8_k16(
        float2 *__restrict__ c_ptr,
        const __half2 *__restrict__ a_ptr,
        const __half2 *__restrict__ b_trans_ptr)
    {
        class I : public Dim<I> { using Dim::Dim; };
        class J : public Dim<J> { using Dim::Dim; };
        class K : public Dim<K> { using Dim::Dim; };

        using A = MMA_M16_K16_F16_A<I, K>;
        using B = MMA_N8_K16_F16_B<K, J>;
        using C = MMA_M16_N8_F32_C<I, J>;

        int lane = threadIdx.x % 32;

        A a;
        A::Index a_index(lane);
        for (int f = 0; f < a.size(); ++f) {
            a(f) = a_ptr[a_index.get<I>(f).get() * 8 + a_index.get<Fold<K, 2>>(f).get()];
        }

        B b;
        B::Index b_index(lane);
        for (int f = 0; f < b.size(); ++f) {
            b(f) = b_trans_ptr[b_index.get<J>(f).get() * 8 + b_index.get<Fold<K, 2>>(f).get()];
        }

        C c;
        c.zero();

        mma_trans(c, a, b, c);

        C::Index idx(lane);
        for (int f = 0; f < c.size(); ++f) {
            c_ptr[idx.get<I>(f).get() * 4 + idx.get<Fold<J, 2>>(f).get()] = c.fragment(f);
        }
    }

    __global__ void mma_m16_n16_k16(
        float2 *__restrict__ c_ptr,
        const __half2 *__restrict__ a_ptr,
        const __half2 *__restrict__ B_trans)
    {
        class I : public Dim<I> { using Dim::Dim; };
        class J : public Dim<J> { using Dim::Dim; };
        class K : public Dim<K> { using Dim::Dim; };

        using A = MMA_M16_K16_F16_A<I, K>;
        using B = MMA_N16_K16_F16_B<K, J>;
        using C = MMA_M16_N16_F32_C<I, J>;

        int lane = threadIdx.x % 32;

        A a;
        A::Index a_index(lane);
        for (int f = 0; f < a.size(); ++f) {
            a(f) = a_ptr[a_index.get<I>(f).get() * 8 + a_index.get<Fold<K, 2>>(f).get()];
        }

        B b;
        B::Index b_index(lane);
        for (int f = 0; f < b.size(); ++f) {
            b(f) = B_trans[b_index.get<J>(f).get() * 8 + b_index.get<Fold<K, 2>>(f).get()];
        }

        C c;
        c.zero();

        mma_trans(c, a, b, c);

        C::Index idx(lane);

        for (int f = 0; f < c.size(); ++f) {
            c_ptr[idx.get<I>(f).get() * 8 + idx.get<Fold<J, 2>>(f).get()] = c.fragment(f);
        }
    }
}
