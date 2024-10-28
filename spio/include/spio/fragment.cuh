#ifndef SPIO_FRAGMENT_H_
#define SPIO_FRAGMENT_H_

/// Define classes for the matrix fragments used with tensor core arithmetic.
#include <cuda_fp16.h>

namespace spio
{
    /// @brief Base class for half-precision matrix fragments.
    /// @tparam NUM_FRAGMENTS The number of 8x8 matrix fragments.
    template <int NUM_FRAGMENTS>
    class _MMA_F16
    {
    public:
        __device__ constexpr int size() const { return NUM_FRAGMENTS; }

        __device__ __half2 &fragment(int idx) { return _data[idx]; }
        __device__ __half2 fragment(int idx) const { return _data[idx]; }

        __device__ unsigned &reg(int idx = 0) { return reinterpret_cast<unsigned *>(_data)[idx]; }
        __device__ unsigned reg(int idx = 0) const { return reinterpret_cast<const unsigned *>(_data)[idx]; }

        __device__ uint2 &reg2(int idx = 0) { return reinterpret_cast<uint2 *>(_data)[idx]; }
        __device__ uint2 reg2(int idx = 0) const { return reinterpret_cast<const uint2 *>(_data)[idx]; }

        __device__ uint4 &reg4(int idx = 0) { return reinterpret_cast<uint4 *>(_data)[idx]; }
        __device__ uint4 reg4(int idx = 0) const { return reinterpret_cast<const uint4 *>(_data)[idx]; }

        __device__ __half2 &operator()(int idx = 0) { return _data[idx]; }
        __device__ __half2 operator()(int idx = 0) const { return _data[idx]; }

        __device__ void *data() { return _data; }
        __device__ const void *data() const { return _data; }

        __device__ unsigned *array() { return reinterpret_cast<unsigned *>(_data); }
        __device__ const unsigned *array() const { return reinterpret_cast<unsigned *>(_data); }

    private:
        __half2 _data[NUM_FRAGMENTS];
    };

    /// @brief Base class for single-precision matrix fragments.
    /// @tparam NUM_FRAGMENTS The number of 8x8 matrix fragments.
    template <int NUM_FRAGMENTS>
    class _MMA_F32
    {
    public:
        static __device__ constexpr int size() { return NUM_FRAGMENTS; }

        __device__ float2 &fragment(int idx) { return _data[idx]; }
        __device__ float2 fragment(int idx) const { return _data[idx]; }

        __device__ __half2 to_half2(int idx) const { return __float22half2_rn(fragment(idx)); }

        __device__ float &operator()(int idx) { return reinterpret_cast<float *>(_data)[idx]; }
        __device__ float operator()(int idx) const { return reinterpret_cast<const float *>(_data)[idx]; }

        __device__ float *array() { return reinterpret_cast<float *>(_data); }
        __device__ const float *array() const { return reinterpret_cast<const float *>(_data); }

        __device__ float2 &vec2(int idx = 0) { return _data[idx]; }
        __device__ float2 vec2(int idx = 0) const { return _data[idx]; }

        __device__ float4 &vec4(int idx = 0) { return reinterpret_cast<float4 *>(_data)[idx]; }
        __device__ float4 vec4(int idx = 0) const { return reinterpret_cast<const float4 *>(_data)[idx]; }

        /// Set all matrix elements equal to zero.
        __device__ void zero()
        {
            for (int idx = 0; idx < NUM_FRAGMENTS; ++idx)
            {
                _data[idx] = make_float2(0, 0);
            }
        }

        __device__ void fill(float2 value)
        {
            for (int idx = 0; idx < NUM_FRAGMENTS; ++idx)
            {
                _data[idx] = value;
            }
        }

        __device__ void add(float2 value)
        {
            for (int idx = 0; idx < NUM_FRAGMENTS; ++idx)
            {
                _data[idx].x += value.x;
                _data[idx].y += value.y;
            }
        }

    private:
        float2 _data[NUM_FRAGMENTS];
    };

    /// @brief  Template base class for 16-row fp16 matrix fragments for operand A.
    /// @tparam _NUM_FRAGMENTS_K Number of 8-column fragments (i.e. the K-dimension).
    template <int _NUM_FRAGMENTS_K>
    class _MMA_M16_N8_F16_A : public _MMA_F16<2 * _NUM_FRAGMENTS_K>
    {
    public:
        /// @brief The number of matrix fragments in the M-dimension.
        static const int NumFragmentsM = 2;

        /// @brief The number of matrix fragments in the K-dimension.
        static const int NumFragmentsK = _NUM_FRAGMENTS_K;

        static const int NumFragments = NumFragmentsM * NumFragmentsK;

        static const int NumElements = NumFragments;

        /// @brief Return the row held by the given lane and fragment index.
        /// @param lane_id The thread's lane number.
        /// @param m_idx The fragment index in the M-dimension.
        /// @return
        __device__ static constexpr int row(unsigned lane_id, int m_idx) { return static_cast<int>(lane_id / 4) + m_idx * 8; }

        /// @brief Return the column held by the given lane and fragment index.
        /// @param lane_id The thread's lane number.
        /// @param k_idx  The fragment index in the K-dimension.
        /// @return
        __device__ static constexpr int col(unsigned lane_id, int k_idx = 0) { return static_cast<int>(lane_id % 4) * 2 + k_idx * 8; }
    };

    /// @brief  Template base class for 8-column fp16 matrix fragments for operand B.
    /// @tparam _NUM_FRAGMENTS_K Number of 8-row matrix fragments (i.e. the K-dimension).
    template <int _NUM_FRAGMENTS_K>
    class _MMA_M16_N8_F16_B : public _MMA_F16<_NUM_FRAGMENTS_K>
    {
    public:
        static const int NumFragmentsK = _NUM_FRAGMENTS_K;
        static const int NumFragmentsN = 1;
        static const int NumFragments = NumFragmentsK * NumFragmentsN;
        static const int NumElements = NumFragments;

        __device__ static constexpr int row(unsigned lane_id, int k_idx = 0) { return (lane_id % 4) * 2 + k_idx * 8; }
        __device__ static constexpr int col(unsigned lane_id) { return lane_id / 4; }
    };

    /// @brief  C or D matrix with float32 elements for M16_N8_K* matrix multiplication with float32 accumulation.
    /// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    class MMA_M16_N8_F32_C : public _MMA_F32<2>
    {
    public:
        static const int NumFragmentsM = 2;
        static const int NumFragmentsN = 1;
        static const int NumFragments = NumFragmentsM * NumFragmentsN;
        static const int NumElements = NumFragments * 2;

        __device__ static constexpr int row(unsigned lane_id, int m_idx) { return static_cast<int>(lane_id / 4) + m_idx * 8; }
        __device__ static constexpr int col2(unsigned lane_id) { return lane_id % 4; }
        __device__ static constexpr int col(unsigned lane_id) { return col2(lane_id) * 2; }
    };

    /// @brief A matrix with float16 elements for M16_N8_K8 matrix multiplication.
    class MMA_M16_K8_F16_A : public _MMA_M16_N8_F16_A<1>
    {
    public:
        using Vector = uint2;
        MMA_M16_K8_F16_A() = default;
        __device__ Vector &vector() { return *static_cast<Vector *>(data()); }
        __device__ const Vector &vector() const { return *static_cast<const Vector *>(data()); }
        __device__ MMA_M16_K8_F16_A(const Vector &v) { vector() = v; }
    };

    /// @brief A matrix with float16 elements for M16_N8_K16 matrix multiplication.
    class MMA_M16_K16_F16_A : public _MMA_M16_N8_F16_A<2>
    {
    public:
        using Vector = uint4;
        MMA_M16_K16_F16_A() = default;
        __device__ Vector &vector() { return *static_cast<Vector *>(data()); }
        __device__ const Vector &vector() const { return *static_cast<const Vector *>(data()); }
        __device__ MMA_M16_K16_F16_A(const Vector &v) { vector() = v; }
    };

    class MMA_N8_K8_F16_B : public _MMA_M16_N8_F16_B<1>
    {
    public:
        using Vector = unsigned;
        MMA_N8_K8_F16_B() = default;
        __device__ Vector &vector() { return *static_cast<Vector *>(data()); }
        __device__ const Vector &vector() const { return *static_cast<const Vector *>(data()); }
        __device__ MMA_N8_K8_F16_B(const Vector &v) { vector() = v; }
    };

    /// @brief B matrix with float16 elements for M16_N8_K16 matrix multiplication.
    /// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    class MMA_N8_K16_F16_B : public _MMA_M16_N8_F16_B<2>
    {
    public:
        using Vector = uint2;
        __device__ Vector &vector() { return *static_cast<Vector *>(data()); }
        __device__ const Vector &vector() const { return *static_cast<const Vector *>(data()); }
    };
}

#endif
