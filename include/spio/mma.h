#ifndef SPIO_MMA_H_
#define SPIO_MMA_H_

/// Implement tensor core matrix multiply-accumulate.
#include <cuda_fp16.h>

/// Tag a function for CUDA __device__ execution.
#define DEVICE __device__

namespace spio
{
    /// @brief A matrix with float16 elements for M16_N8_K16 matrix multiplication.
    /// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    class MMA_M16_N8_K16_F16_A
    {
    public:
        /// @brief The number of matrix fragments in the M-dimension.
        static const int NumFragmentsM = 2;

        /// @brief The number of matrix fragments in the K-dimension.
        static const int NumFragmentsK = 2;

        static const int NumFragments = NumFragmentsM * NumFragmentsK;

        static const int NumElements = NumFragments;

        /// @brief Return the row held by the given lane and fragment index.
        /// @param lane_id The thread's lane number.
        /// @param m_idx The fragment index in the M-dimension.
        /// @return
        DEVICE static constexpr int row(int lane_id, int m_idx) { return lane_id / 4 + m_idx * 8; }

        /// @brief Return the column held by the given lane and fragment index.
        /// @param lane_id The thread's lane number.
        /// @param k_idx  The fragment index in the K-dimension.
        /// @return
        DEVICE static constexpr int col(int lane_id, int k_idx) { return (lane_id % 4) * 2 + k_idx * 8; }

        DEVICE __half2 &fragment(int idx) { return _data[idx]; }
        DEVICE __half2 fragment(int idx) const { return _data[idx]; }

        DEVICE unsigned &reg(int idx) { return reinterpret_cast<unsigned *>(_data)[idx]; }
        DEVICE unsigned reg(int idx) const { return reinterpret_cast<const unsigned *>(_data)[idx]; }

        DEVICE __half2 &operator()(int idx) { return _data[idx]; }
        DEVICE __half2 operator()(int idx) const { return _data[idx]; }

    private:
        __half2 _data[NumFragments];
    };

    /// @brief B matrix with float16 elements for M16_N8_K16 matrix multiplication.
    /// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    class MMA_M16_N8_K16_F16_B
    {
    public:
        static const int NumFragmentsK = 2;
        static const int NumFragmentsN = 1;
        static const int NumFragments = NumFragmentsK * NumFragmentsN;
        static const int NumElements = NumFragments;

        DEVICE static constexpr int row(int lane_id, int k_idx) { return (lane_id % 4) * 2 + k_idx * 8; }
        DEVICE static constexpr int col(int lane_id) { return lane_id / 4; }

        DEVICE __half2 &fragment(int idx) { return _data[idx]; }
        DEVICE __half2 fragment(int idx) const { return _data[idx]; }

        DEVICE unsigned &reg(int idx) { return reinterpret_cast<unsigned *>(_data)[idx]; }
        DEVICE unsigned reg(int idx) const { return reinterpret_cast<const unsigned *>(_data)[idx]; }

        DEVICE __half2 &operator()(int idx) { return _data[idx]; }
        DEVICE __half2 operator()(int idx) const { return _data[idx]; }

    private:
        __half2 _data[NumFragments];
    };

    /// @brief  C or D matrix with float32 elements for M16_N8_K16 matrix multiplication with float32 accumulation.
    /// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    class MMA_M16_N8_K16_F32_C
    {
    public:
        static const int NumFragmentsM = 2;
        static const int NumFragmentsN = 1;
        static const int NumFragments = NumFragmentsM * NumFragmentsN;
        static const int NumElements = NumFragments * 2;

        DEVICE static constexpr int row(int lane_id, int m_idx) { return lane_id / 4 + m_idx * 8; }
        DEVICE static constexpr int col(int lane_id) { return (lane_id % 4) * 2; }

        DEVICE float2 &fragment(int idx) { return _data[idx]; }
        DEVICE float2 fragment(int idx) const { return _data[idx]; }

        DEVICE float &operator()(int idx) { return reinterpret_cast<float *>(_data)[idx]; }
        DEVICE float operator()(int idx) const { return reinterpret_cast<const float *>(_data)[idx]; }

        DEVICE void zero()
        {
            for (int idx = 0; idx < NumFragments; ++idx)
            {
                _data[idx] = make_float2(0, 0);
            }
        }

    private:
        float2 _data[NumFragments];
    };

    /// @brief Perform D = A x B + C matrix-multiplication with matrix fragments.
    /// @param d Output matrix in float32
    /// @param a Input matrix A in float16.
    /// @param b Input matrix B in float16.
    /// @param c Accumulate fragment in float32.
    /// https://docs.nvidia.com/cuda/parallel-thread-execution/#multiply-and-accumulate-instruction-mma
    DEVICE void mma(
        MMA_M16_N8_K16_F32_C &d,
        const MMA_M16_N8_K16_F16_A &a,
        const MMA_M16_N8_K16_F16_B &b,
        const MMA_M16_N8_K16_F32_C &c)
    {
        // mma.sync.aligned.m16n8k16.row.col.dtype.f16.f16.ctype d, a, b, c;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            " {%0, %1, %2, %3},"
            " {%4, %5, %6, %7},"
            " {%8, %9}, "
            " {%10, %11, %12, %13};"
            : "=f"(d(0)), "=f"(d(1)), "=f"(d(2)), "=f"(d(3))
            : "r"(a.reg(0)), "r"(a.reg(1)), "r"(a.reg(2)), "r"(a.reg(3)),
              "r"(b.reg(0)), "r"(b.reg(1)),
              "f"(c(0)), "f"(c(1)), "f"(c(2)), "f"(c(3)));
    }

}

#endif
