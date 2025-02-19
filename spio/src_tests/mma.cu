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
        using A = MMA_M16_K8_F16_A;
        using B = MMA_N8_K8_F16_B;
        using C = MMA_M16_N8_F32_C;

        int lane = threadIdx.x % 32;

        A a;
        A::Index a_index(lane);

        for (int f = 0; f < a.size(); ++f) {
            a(f) = a_ptr[a_index.i(f) * 4 + a_index.k2(f)];
        }

        B b;
        B::Index b_index(lane);
        b() = b_trans_ptr[b_index.j() * 4 + b_index.k2()];

        C c;
        c.zero();

        matmul_trans(c, a, b, c);

        C::Index idx(lane);       
        c_ptr[idx.i(0) * 4 + idx.j2(0)] = c.fragment(0);
        c_ptr[idx.i(1) * 4 + idx.j2(1)] = c.fragment(1);
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
        using A = MMA_M16_K16_F16_A;
        using B = MMA_N8_K16_F16_B;
        using C = MMA_M16_N8_F32_C;

        int lane = threadIdx.x % 32;

        A a;
        A::Index a_index(lane);
        for (int f = 0; f < 4; ++f) {
            a(f) = a_ptr[a_index.i(f) * 8 + a_index.k2(f)];
        }

        B b;
        B::Index b_index(lane);
        for (int f = 0; f < 2; ++f) {
            b(f) = b_trans_ptr[b_index.j() * 8 + b_index.k2(f)];
        }

        C c;
        c.zero();

        matmul_trans(c, a, b, c);

        C::Index idx(lane);
        for (int f = 0; f < c.size(); ++f) {
            c_ptr[idx.i(f) * 4 + idx.j2(f)] = c.fragment(f);
        }
    }

    __global__ void mma_m16_n16_k16(
        float2 *__restrict__ c_ptr,
        const __half2 *__restrict__ a_ptr,
        const __half2 *__restrict__ B_trans)
    {
        using A = MMA_M16_K16_F16_A;
        using B = MMA_N16_K16_F16_B;
        using C = MMA_M16_N16_F32_C;

        int lane = threadIdx.x % 32;

        A a;
        A::Index a_index(lane);
        for (int f = 0; f < a.size(); ++f) {
            a(f) = a_ptr[a_index.i(f) * 8 + a_index.k2(f)];
        }

        B b;
        B::Index b_index(lane);
        for (int f = 0; f < b.size(); ++f) {
            b(f) = B_trans[b_index.j(f) * 8 + b_index.k2(f)];
        }

        C c;
        c.zero();

        matmul_trans(c, a, b, c);

        C::Index idx(lane);

        for (int f = 0; f < c.size(); ++f) {
            c_ptr[idx.i(f) * 8 + idx.j2(f)] = c.fragment(f);
        }
    }
}
