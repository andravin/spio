#ifndef SPIO_FRAGMENT_H_
#define SPIO_FRAGMENT_H_

/// Define classes for the matrix fragments used with tensor core arithmetic.
#include <cuda_fp16.h>

#include "spio/fragment_index.h"
#include "spio/fragment_load_index.h"
#include "spio/ldmatrix.cuh"
#include "spio/tensor.h"

namespace spio {

    /// @brief CRTP mixin that provides tensor-like access for fragment classes.
    /// Derived classes must define `tensor_type` and provide `data()` method.
    template <typename Derived> class FragmentTensorAccess {
    public:
        __device__ auto as_tensor() {
            using tensor_type = typename Derived::tensor_type;
            return tensor_type(static_cast<Derived*>(this)->data());
        }

        __device__ auto as_tensor() const {
            using tensor_type = typename Derived::tensor_type;
            using data_type = typename tensor_type::data_type;
            return tensor_type(const_cast<data_type*>(static_cast<const Derived*>(this)->data()));
        }

        // Subscript operators forwarding to tensor
        template <typename DimType> __device__ auto operator[](DimType idx) {
            return as_tensor()[idx];
        }

        template <typename DimType> __device__ auto operator[](DimType idx) const {
            return as_tensor()[idx];
        }

        // Subscript with Coordinates
        template <typename... Dims> __device__ auto operator[](const Coordinates<Dims...>& coords) {
            return as_tensor()[coords];
        }

        template <typename... Dims>
        __device__ auto operator[](const Coordinates<Dims...>& coords) const {
            return as_tensor()[coords];
        }
    };

    template <int NUM_FRAGMENTS, typename T> class alignas(8) _MMA {
    public:
        using data_type = T;

        __device__ static constexpr int size() {
            return NUM_FRAGMENTS;
        }

        __device__ T& fragment(int idx) {
            return _data[idx];
        }

        __device__ T fragment(int idx) const {
            return _data[idx];
        }

        __device__ T* data() {
            return _data;
        }

        __device__ const T* data() const {
            return _data;
        }

        __device__ void fill(T value) {
            for (int idx = 0; idx < NUM_FRAGMENTS; ++idx) {
                _data[idx] = value;
            }
        }

        /// Set all matrix elements equal to zero.
        __device__ void zero() {
            fill(T{0});
        }

        __device__ void add(T value) {
            for (int idx = 0; idx < NUM_FRAGMENTS; ++idx) {
                _data[idx] += value;
            }
        }

    private:
        T _data[NUM_FRAGMENTS];
    };

    /// @brief Base class for half-precision matrix fragments.
    /// @tparam NUM_FRAGMENTS The number of 8x8 matrix fragments.
    template <int NUM_FRAGMENTS> class _MMA_F16 : public _MMA<NUM_FRAGMENTS, __half2> {
    public:
        using Base = _MMA<NUM_FRAGMENTS, __half2>;
        using Base::data;
        using Base::fragment;

        __device__ unsigned& reg(int idx = 0) {
            return reinterpret_cast<unsigned*>(data())[idx];
        }

        __device__ unsigned reg(int idx = 0) const {
            return reinterpret_cast<const unsigned*>(data())[idx];
        }

        __device__ uint2& reg2(int idx = 0) {
            return reinterpret_cast<uint2*>(data())[idx];
        }

        __device__ uint2 reg2(int idx = 0) const {
            return reinterpret_cast<const uint2*>(data())[idx];
        }

        __device__ uint4& reg4(int idx = 0) {
            return reinterpret_cast<uint4*>(data())[idx];
        }

        __device__ uint4 reg4(int idx = 0) const {
            return reinterpret_cast<const uint4*>(data())[idx];
        }

        __device__ __half2& operator()(int idx = 0) {
            return fragment(idx);
        }

        __device__ __half2 operator()(int idx = 0) const {
            return fragment(idx);
        }

        __device__ unsigned* array() {
            return reinterpret_cast<unsigned*>(data());
        }

        __device__ const unsigned* array() const {
            return reinterpret_cast<unsigned*>(data());
        }
    };

    /// @brief Base class for single-precision matrix fragments.
    /// @tparam NUM_FRAGMENTS The number of 8x8 matrix fragments.
    template <int NUM_FRAGMENTS> class _MMA_F32 : public _MMA<NUM_FRAGMENTS, float2> {
    public:
        using Base = _MMA<NUM_FRAGMENTS, float2>;
        using Base::data;
        using Base::fragment;

        __device__ __half2 to_half2(int idx) const {
            return __float22half2_rn(fragment(idx));
        }

        __device__ float& operator()(int idx) {
            return reinterpret_cast<float*>(data())[idx];
        }

        __device__ float operator()(int idx) const {
            return reinterpret_cast<const float*>(data())[idx];
        }

        __device__ float* array() {
            return reinterpret_cast<float*>(data());
        }

        __device__ const float* array() const {
            return reinterpret_cast<const float*>(data());
        }

        __device__ float2& vec2(int idx = 0) {
            return data()[idx];
        }

        __device__ float2 vec2(int idx = 0) const {
            return data()[idx];
        }

        __device__ float4& vec4(int idx = 0) {
            return reinterpret_cast<float4*>(data())[idx];
        }

        __device__ float4 vec4(int idx = 0) const {
            return reinterpret_cast<const float4*>(data())[idx];
        }

        __device__ void to_half2(_MMA_F16<NUM_FRAGMENTS>& dst) const {
            for (int i = 0; i < NUM_FRAGMENTS; ++i) {
                dst.fragment(i) = fragment(i);
            }
        }
    };

    /// @brief  Template base class for 16-row fp16 matrix fragments for operand A.
    /// @tparam _NUM_FRAGMENTS_K Number of 8-column fragments (i.e. the K-dimension).
    template <typename RowDim, typename ColDim, int _NUM_FRAGMENTS_K>
    class _MMA_M16_N8_F16_A : public _MMA_F16<2 * _NUM_FRAGMENTS_K> {
    public:
        using compound_index_type = MMA_A_88_F16_Index<RowDim, ColDim>;
        using Base = _MMA_F16<2 * _NUM_FRAGMENTS_K>;
        using typename Base::data_type;

        using row_dim = RowDim;
        using col_dim = ColDim;
    };

    /// @brief  Template base class for 8-column fp16 matrix fragments for operand B.
    /// @tparam _NUM_FRAGMENTS_K Number of 8-row matrix fragments (i.e. the K-dimension).
    template <typename RowDim, typename ColDim, int _NUM_FRAGMENTS_K>
    class _MMA_M16_N8_F16_B : public _MMA_F16<_NUM_FRAGMENTS_K> {
    public:
        using compound_index_type = MMA_B_88_F16_Index<RowDim, ColDim>;
        using Base = _MMA_F16<_NUM_FRAGMENTS_K>;
        using typename Base::data_type;

        using row_dim = RowDim;
        using col_dim = ColDim;
    };

    /// @brief  C or D matrix with float32 elements for M16_N8_K* matrix multiplication with float32
    /// accumulation.
    /// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    template <typename RowDim, typename ColDim>
    class MMA_M16_N8_F32_C : public _MMA_F32<2>,
                             public FragmentTensorAccess<MMA_M16_N8_F32_C<RowDim, ColDim>> {
    public:
        using compound_index_type = MMA_C_88_F32_Index<RowDim, ColDim>;
        using row_dim = RowDim;
        using col_dim = ColDim;
        using Base = _MMA_F32<2>;
        using Base::data_type;

        // Tensor type: 2x1 fragments, column-major (row varies fastest)
        using tensor_type =
            Tensor<data_type, DimInfo<Fold<ColDim, 8>, 1, 2>, DimInfo<Fold<RowDim, 8>, 2, 1>>;
    };

    template <typename RowDim, typename ColDim>
    class MMA_M16_N16_F32_C : public _MMA_F32<4>,
                              public FragmentTensorAccess<MMA_M16_N16_F32_C<RowDim, ColDim>> {
    public:
        using compound_index_type = MMA_C_88_F32_Index<RowDim, ColDim>;
        using row_dim = RowDim;
        using col_dim = ColDim;
        using Base = _MMA_F32<4>;
        using Base::data_type;

        // Tensor type: 2x2 fragments, column-major (row varies fastest)
        using tensor_type =
            Tensor<data_type, DimInfo<Fold<ColDim, 8>, 2, 2>, DimInfo<Fold<RowDim, 8>, 2, 1>>;
    };

    /// @brief  Mixin class for loading fragments from memory.
    /// @tparam Derived The derived fragment class that provides load() and load_trans() methods.
    template <typename Derived> class FragmentLoader {
    public:
        /// @brief Load a fragment from memory.
        /// @param p The pointer to the memory to load the fragment from.
        /// @return The loaded fragment.
        static __device__ Derived load_new(const void* p) {
            Derived ret;
            ret.load(p);
            return ret;
        }

        /// @brief Load a transposed fragment from memory.
        /// @param p The pointer to the memory to load the fragment from.
        /// @return The loaded fragment.
        static __device__ Derived load_trans_new(const void* p) {
            Derived ret;
            ret.load_trans(p);
            return ret;
        }
    };

    /// @brief A matrix with float16 elements for M16_N8_K8 matrix multiplication.
    template <typename RowDim, typename ColDim>
    class MMA_M16_K8_F16_A : public _MMA_M16_N8_F16_A<RowDim, ColDim, 1>,
                             public FragmentLoader<MMA_M16_K8_F16_A<RowDim, ColDim>>,
                             public FragmentTensorAccess<MMA_M16_K8_F16_A<RowDim, ColDim>> {
    public:
        using vector_type = uint2;
        using load_index_type = MMA_A_M16_K8_F16_LoadIndex<RowDim, ColDim>;
        using Base = _MMA_M16_N8_F16_A<RowDim, ColDim, 1>;
        using Base::data;
        using typename Base::data_type;

        // Tensor type: 2x1 fragments (16 rows × 8 cols), column-major (row varies fastest)
        using tensor_type =
            Tensor<data_type, DimInfo<Fold<ColDim, 8>, 1, 2>, DimInfo<Fold<RowDim, 8>, 2, 1>>;

        MMA_M16_K8_F16_A() = default;

        __device__ vector_type& vector() {
            return *reinterpret_cast<vector_type*>(data());
        }

        __device__ const vector_type& vector() const {
            return *reinterpret_cast<const vector_type*>(data());
        }

        __device__ MMA_M16_K8_F16_A(const vector_type& v) {
            vector() = v;
        }

        __device__ void load(const void* p) {
            vector() = ldmatrix_x2(p);
        }

        __device__ void load_trans(const void* p) {
            vector() = ldmatrix_x2_trans(p);
        }
    };

    /// @brief A matrix with float16 elements for M16_N8_K16 matrix multiplication.
    template <typename RowDim, typename ColDim>
    class MMA_M16_K16_F16_A : public _MMA_M16_N8_F16_A<RowDim, ColDim, 2>,
                              public FragmentLoader<MMA_M16_K16_F16_A<RowDim, ColDim>>,
                              public FragmentTensorAccess<MMA_M16_K16_F16_A<RowDim, ColDim>> {
    public:
        using vector_type = uint4;
        using load_index_type = MMA_A_M16_K16_F16_LoadIndex<RowDim, ColDim>;
        using Base = _MMA_M16_N8_F16_A<RowDim, ColDim, 2>;
        using Base::data;
        using typename Base::data_type;

        // Tensor type: 2x2 fragments (16 rows × 16 cols), column-major (row varies fastest)
        using tensor_type =
            Tensor<data_type, DimInfo<Fold<ColDim, 8>, 2, 2>, DimInfo<Fold<RowDim, 8>, 2, 1>>;

        MMA_M16_K16_F16_A() = default;

        __device__ vector_type& vector() {
            return *reinterpret_cast<vector_type*>(data());
        }

        __device__ const vector_type& vector() const {
            return *reinterpret_cast<const vector_type*>(data());
        }

        __device__ MMA_M16_K16_F16_A(const vector_type& v) {
            vector() = v;
        }

        __device__ void load(const void* p) {
            vector() = ldmatrix_x4(p);
        }

        __device__ void load_trans(const void* p) {
            vector() = ldmatrix_x4_trans(p);
        }
    };

    template <typename RowDim, typename ColDim>
    class MMA_N8_K8_F16_B : public _MMA_M16_N8_F16_B<RowDim, ColDim, 1>,
                            public FragmentLoader<MMA_N8_K8_F16_B<RowDim, ColDim>>,
                            public FragmentTensorAccess<MMA_N8_K8_F16_B<RowDim, ColDim>> {
    public:
        using vector_type = unsigned;
        using load_index_type = MMA_B_N8_K8_F16_LoadIndex<RowDim, ColDim>;
        using Base = _MMA_M16_N8_F16_B<RowDim, ColDim, 1>;
        using Base::data;
        using typename Base::data_type;

        // Tensor type: 1x1 fragment (8 cols × 8 rows)
        using tensor_type =
            Tensor<data_type, DimInfo<Fold<ColDim, 8>, 1, 1>, DimInfo<Fold<RowDim, 8>, 1, 1>>;

        MMA_N8_K8_F16_B() = default;

        __device__ vector_type& vector() {
            return *reinterpret_cast<vector_type*>(data());
        }

        __device__ const vector_type& vector() const {
            return *reinterpret_cast<const vector_type*>(data());
        }

        __device__ MMA_N8_K8_F16_B(const vector_type& v) {
            vector() = v;
        }

        __device__ void load(const void* p) {
            vector() = ldmatrix_x1(p);
        }

        __device__ void load_trans(const void* p) {
            vector() = ldmatrix_x1_trans(p);
        }
    };

    /// @brief B matrix with float16 elements for M16_N8_K16 matrix multiplication.
    /// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    template <typename RowDim, typename ColDim>
    class MMA_N8_K16_F16_B : public _MMA_M16_N8_F16_B<RowDim, ColDim, 2>,
                             public FragmentLoader<MMA_N8_K16_F16_B<RowDim, ColDim>>,
                             public FragmentTensorAccess<MMA_N8_K16_F16_B<RowDim, ColDim>> {
    public:
        using vector_type = uint2;
        using load_index_type = MMA_B_N16_K16_F16_LoadIndex<RowDim, ColDim>;
        using Base = _MMA_M16_N8_F16_B<RowDim, ColDim, 2>;
        using Base::data;
        using typename Base::data_type;

        // Tensor type: 1x2 fragments (8 cols × 16 rows), column-major (K/row varies fastest)
        using tensor_type =
            Tensor<data_type, DimInfo<Fold<ColDim, 8>, 1, 2>, DimInfo<Fold<RowDim, 8>, 2, 1>>;

        MMA_N8_K16_F16_B() = default;

        __device__ vector_type& vector() {
            return *reinterpret_cast<vector_type*>(data());
        }

        __device__ const vector_type& vector() const {
            return *reinterpret_cast<const vector_type*>(data());
        }

        __device__ void load(const void* p) {
            vector() = ldmatrix_x2(p);
        }

        __device__ void load_trans(const void* p) {
            vector() = ldmatrix_x2_trans(p);
        }
    };

    template <typename RowDim, typename ColDim>
    class MMA_N16_K16_F16_B : public _MMA_M16_N8_F16_B<RowDim, ColDim, 4>,
                              public FragmentLoader<MMA_N16_K16_F16_B<RowDim, ColDim>>,
                              public FragmentTensorAccess<MMA_N16_K16_F16_B<RowDim, ColDim>> {
    public:
        using vector_type = uint4;
        using load_index_type = MMA_B_N16_K16_F16_LoadIndex<RowDim, ColDim>;
        using Base = _MMA_M16_N8_F16_B<RowDim, ColDim, 4>;
        using Base::data;
        using typename Base::data_type;

        // Tensor type: 2x2 fragments (16 cols × 16 rows), column-major (K/row varies fastest)
        using tensor_type =
            Tensor<data_type, DimInfo<Fold<ColDim, 8>, 2, 2>, DimInfo<Fold<RowDim, 8>, 2, 1>>;

        MMA_N16_K16_F16_B() = default;

        __device__ vector_type& vector() {
            return *reinterpret_cast<vector_type*>(data());
        }

        __device__ const vector_type& vector() const {
            return *reinterpret_cast<const vector_type*>(data());
        }

        __device__ void load(const void* p) {
            vector() = ldmatrix_x4(p);
        }

        __device__ void load_trans(const void* p) {
            vector() = ldmatrix_x4_trans(p);
        }
    };

}

#endif
