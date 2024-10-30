/// Functions that implement the ldmatrix PTX instruction.
///
/// Includes functions for loading 1, 2, or 4 matrix fragments
/// of size 8x8 and data-type fp16 from shared memory to registers.
#ifndef SPIO_LDMATRIX_H_
#define SPIO_LDMATRIX_H_

namespace spio
{
    /// @brief Load a single 8x8 fp16 matrix fragment from shared memory.
    /// @details https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix
    /// @param p Pointers to the rows of the matrix (in threads 0-7).
    /// @return The matrix fragment as one warp-wide register.
    __device__ unsigned ldmatrix_x1(const void *p)
    {
        size_t s = __cvta_generic_to_shared(p);
        unsigned x;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16"
            " {%0}, [%1];"
            : "=r"(x)
            : "l"(s));
        return x;
    }

    /// @brief Load two 8x8 fp16 matrix fragments from shared memory.
    /// @param p Pointers to the rows of the matrix (in threads 0-15).
    /// @return The two matrix fragments as two warp-wide registers.
    __device__ uint2 ldmatrix_x2(const void *p)
    {
        size_t s = __cvta_generic_to_shared(p);
        uint2 v;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16"
            " {%0, %1}, [%2];"
            : "=r"(v.x), "=r"(v.y)
            : "l"(s));
        return v;
    }

    /// @brief  Load four 8x8 fp16 matrix fragments from shared memory.
    /// @param p Pointers to the rows of the matrices (in threads 0-31).
    /// @return The four matrix fragments as two warp-wide registers.
    __device__ uint4 ldmatrix_x4(const void *p)
    {
        size_t s = __cvta_generic_to_shared(p);
        uint4 v;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            " {%0, %1, %2, %3}, [%4];"
            : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
            : "l"(s));
        return v;
    }

    /// @brief Load a single transposed 8x8 fp16 matrix fragment from shared memory.
    /// @param p Pointers to the rows of the matrix (in threads 0-7).
    /// @return The matrix fragment as one warp-wide register.
    __device__ unsigned ldmatrix_x1_trans(const void *p)
    {
        size_t s = __cvta_generic_to_shared(p);
        unsigned x;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16"
            " {%0}, [%1];"
            : "=r"(x)
            : "l"(s));
        return x;
    }

    /// @brief Load two transposed 8x8 fp16 matrix fragments from shared memory.
    /// @param p Pointers to the rows of the matrix (in threads 0-15).
    /// @return The two matrix fragments as two warp-wide registers.
    __device__ uint2 ldmatrix_x2_trans(const void *p)
    {
        size_t s = __cvta_generic_to_shared(p);
        uint2 v;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"
            " {%0, %1}, [%2];"
            : "=r"(v.x), "=r"(v.y)
            : "l"(s));
        return v;
    }

    /// @brief  Load four transposed 8x8 fp16 matrix fragments from shared memory.
    /// @param p Pointers to the rows of the matrices (in threads 0-31).
    /// @return The four matrix fragments as four warp-wide registers.
    __device__ uint4 ldmatrix_x4_trans(const void *p)
    {
        size_t s = __cvta_generic_to_shared(p);
        uint4 v;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            " {%0, %1, %2, %3}, [%4];"
            : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
            : "l"(s));
        return v;
    }
}

#endif
