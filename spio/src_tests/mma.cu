#include "spio/mma.cuh"

extern "C"
{
    using namespace spio;

    /// @brief  Test mma.m16n8k8 with float16 data.
    /// @param A 16m x 8k matrix with float16 elements.
    /// @param B_trans 8n x 8k matrix with float16 elements.
    /// @param C 16m x 8n matrix with float32 elements.
    __global__ void mma_m16_n8_k8(
        float2 *__restrict__ C,
        const __half2 *__restrict__ A,
        const __half2 *__restrict__ B_trans)
    {
        int lane = threadIdx.x % 32;

        MMA_M16_K8_F16_A a;
        a(0) = A[a.row(lane, 0) * 4 + a.col(lane) / 2];
        a(1) = A[a.row(lane, 1) * 4 + a.col(lane) / 2];

        MMA_N8_K8_F16_B b;
        b() = B_trans[b.col(lane) * 4 + b.row(lane) / 2];

        MMA_M16_N8_F32_C c;
        c.zero();

        mma_m16_n8_k8(c.vec4(), a.reg2(), b.reg(), c.vec4());

        C[c.row(lane, 0) * 4 + c.col(lane) / 2] = c.fragment(0);
        C[c.row(lane, 1) * 4 + c.col(lane) / 2] = c.fragment(1);
    }

    /// @brief  Test mma.m16n8k16 with float16 data.
    /// @param A 16m x 16k matrix with float16 elements.
    /// @param B_trans 8n x 16k matrix with float16 elements.
    /// @param C 16m x 8n matrix with float32 elements.
    __global__ void mma_m16_n8_k16(
        float2 *__restrict__ C,
        const __half2 *__restrict__ A,
        const __half2 *__restrict__ B_trans)
    {
        int lane = threadIdx.x % 32;

        MMA_M16_K16_F16_A a;
        a(0) = A[a.row(lane, 0) * 8 + a.col(lane, 0) / 2];
        a(1) = A[a.row(lane, 1) * 8 + a.col(lane, 0) / 2];
        a(2) = A[a.row(lane, 0) * 8 + a.col(lane, 1) / 2];
        a(3) = A[a.row(lane, 1) * 8 + a.col(lane, 1) / 2];

        MMA_N8_K16_F16_B b;
        b(0) = B_trans[b.col(lane) * 8 + b.row(lane, 0) / 2];
        b(1) = B_trans[b.col(lane) * 8 + b.row(lane, 1) / 2];

        MMA_M16_N8_F32_C c;
        c.zero();

        mma_m16_n8_k16(c.vec4(), a.reg4(), b.reg2(), c.vec4());

        C[c.row(lane, 0) * 4 + c.col(lane) / 2] = c.fragment(0);
        C[c.row(lane, 1) * 4 + c.col(lane) / 2] = c.fragment(1);
    }
}
