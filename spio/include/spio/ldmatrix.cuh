/// Functions that implement the ldmatrix PTX instruction.
///
/// Includes functions for loading 1, 2, or 4 matrix fragments
/// of size 8x8 and data-type fp16 from shared memory to registers.
#ifndef SPIO_LDMATRIX_H_
#define SPIO_LDMATRIX_H_

namespace spio
{
    /// Loads a single 8x8 fp16 matrix fragment from shared memory.
    ///
    /// Parameters:
    ///   p   Pointers to the rows of the matrix (in threads 0-7).
    ///
    /// Returns:
    ///   The matrix fragment as one warp-wide register.
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

    /// Loads two 8x8 fp16 matrix fragments from shared memory.
    ///
    /// Parameters:
    ///   p   Pointers to the rows of the matrix (in threads 0-15).
    ///
    /// Returns:
    ///   The two matrix fragments as two warp-wide registers.
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

    /// Loads four 8x8 fp16 matrix fragments from shared memory.
    ///
    /// Parameters:
    ///   p   Pointers to the rows of the matrices (in threads 0-31).
    ///
    /// Returns:
    ///   The four matrix fragments as four warp-wide registers.
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

    /// Loads a single transposed 8x8 fp16 matrix fragment from shared memory.
    ///
    /// Parameters:
    ///   p   Pointers to the rows of the matrix (in threads 0-7).
    ///
    /// Returns:
    ///   The matrix fragment as one warp-wide register.
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

    /// Loads two transposed 8x8 fp16 matrix fragments from shared memory.
    ///
    /// Parameters:
    ///   p   Pointers to the rows of the matrix (in threads 0-15).
    ///
    /// Returns:
    ///   The two matrix fragments as two warp-wide registers.
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

    /// Loads four transposed 8x8 fp16 matrix fragments from shared memory.
    ///
    /// Parameters:
    ///   p   Pointers to the rows of the matrices (in threads 0-31).
    ///
    /// Returns:
    ///   The four matrix fragments as four warp-wide registers.
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
