#ifndef SPIO_MMA_H_
#define SPIO_MMA_H_

/// Define functions for tensor core matrix multiply-accumulate.

namespace spio
{
    /// @brief Perform D = A x B + C matrix-multiplication with tensor cores.
    /// @param d Output matrix in float32
    /// @param a Input matrix A is a 16 x 16 fragmentin float16.
    /// @param b Input matrix B is an 16 x 8 fragment float16.
    /// @param c Accumulate matrix C is a 16 x 8 fragment in float32.
    /// https://docs.nvidia.com/cuda/parallel-thread-execution/#multiply-and-accumulate-instruction-mma
    __device__ void mma_m16_n8_k16(
        float4 &d,
        uint4 a,
        uint2 b,
        float4 c)
    {
        // mma.sync.aligned.m16n8k16.row.col.dtype.f16.f16.ctype d, a, b, c;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            " {%0, %1, %2, %3},"
            " {%4, %5, %6, %7},"
            " {%8, %9}, "
            " {%10, %11, %12, %13};"
            : "=f"(d.x), "=f"(d.y), "=f"(d.z), "=f"(d.w)
            : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
              "r"(b.x), "r"(b.y),
              "f"(c.x), "f"(c.y), "f"(c.z), "f"(c.w));
    }

    /// @brief Perform  D= A x B + C matrix-multiplication with tensor cores.
    /// @param d Output matrix is a 16 x 8 fragment in float32.
    /// @param a Input matrix A is a 16 x 8 fragment in float16.
    /// @param b Input matrix B is an 8 x 8 fragment in float16.
    /// @param c Accumulate matrix C is a 16 x 8 fragment in float32.
    /// @return
    __device__ void mma_m16_n8_k8(
        float4 &d,
        uint2 a,
        unsigned b,
        float4 c)
    {
        // mma.sync.aligned.m16n8k8.row.col.dtype.f16.f16.ctype d, a, b, c;
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            " {%0, %1, %2, %3},"
            " {%4, %5},"
            " {%6}, "
            " {%7, %8, %9, %10};"
            : "=f"(d.x), "=f"(d.y), "=f"(d.z), "=f"(d.w)
            : "r"(a.x), "r"(a.y),
              "r"(b),
              "f"(c.x), "f"(c.y), "f"(c.z), "f"(c.w));
    }
}

#endif
