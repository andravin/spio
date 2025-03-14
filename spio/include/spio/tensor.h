#ifndef SPIO_TENSOR_H_
#define SPIO_TENSOR_H_

#include "spio/macros.h"

namespace spio
{
    /// A base class for all Tensor* classes.
    template <typename _data_type>
    class TensorBase
    {
    public:
        using data_type = _data_type;
        static constexpr int element_size = sizeof(_data_type);
        DEVICE constexpr TensorBase(_data_type *data = nullptr) : _data(data) {}
        DEVICE constexpr _data_type *get() const { return _data; }
        DEVICE void reset(_data_type *data) { _data = data; }
        DEVICE constexpr _data_type &operator*() const { return *_data; }
        DEVICE constexpr _data_type *operator->() const { return _data; }

    private:
        _data_type *_data;
    };

    template <typename _data_type, class _dim_0, int _size_0, int _stride_0 = 1>
    class Tensor1D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;

        using dim_0 = _dim_0;
        static constexpr dim_0 size_0 = _size_0;
        static constexpr int stride_0 = _stride_0;
        static constexpr int size = size_0.get();

        DEVICE constexpr Tensor1D operator[](dim_0 d0) const { return Tensor1D(get() + d0.get() * stride_0); }

        template <int _size>
        DEVICE constexpr Tensor1D<data_type, dim_0, size_0.get() - _size, stride_0>
        slice(dim_0 slice_start)
        {
            return Tensor1D<data_type, dim_0, size_0.get() - _size, stride_0>((*this)[slice_start].get());
        }
    };

    /// A base class for a 2-dimensional index.
    template <
        typename _data_type,
        class _dim_0,
        class _dim_1,
        int _size_0,
        int _size_1,
        int _stride_1 = 1,
        int _stride_0 = _size_1 * _stride_1>
    class Tensor2D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;

        using dim_0 = _dim_0;
        using dim_1 = _dim_1;
        static constexpr dim_0 size_0 = _size_0;
        static constexpr dim_1 size_1 = _size_1;
        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;
        static constexpr int size = size_0.get() * size_1.get();

        DEVICE constexpr Tensor2D operator[](dim_0 d0) const { return Tensor2D(get() + d0.get() * stride_0); }
        DEVICE constexpr Tensor2D operator[](dim_1 d1) const { return Tensor2D(get() + d1.get() * stride_1); }

        template <int _size>
        DEVICE constexpr Tensor2D<data_type, dim_0, dim_1, size_0.get() - _size, size_1, stride_1, stride_0>
        slice(dim_0 slice_start)
        {
            return Tensor2D<data_type, dim_0, dim_1, size_0.get() - _size, size_1, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr Tensor2D<data_type, dim_0, dim_1, size_0, size_1.get() - _size, stride_1, stride_0>
        slice(dim_1 slice_start)
        {
            return Tensor2D<data_type, dim_0, dim_1, size_0, size_1.get() - _size, stride_1, stride_0>((*this)[slice_start].get());
        }
    };

    /// A base class for a 3-dimensional index.
    // template <typename _data_type,
    //           int _D1, int _D2,
    //           int _D2_Stride = 1, int _D1_Stride = _D2 * _D2_Stride, int _D0_Stride = _D1 * _D1_Stride>
    // class Tensor3D : public TensorBase<_data_type>
    // {
    // public:
    //     using TensorBase<_data_type>::TensorBase;
    //     using TensorBase<_data_type>::get;

    //     DEVICE constexpr Tensor3D _d2(int d2) const { return Tensor3D(get() + d2 * _D2_Stride); }
    //     DEVICE constexpr Tensor3D _d1(int d1) const { return Tensor3D(get() + d1 * _D1_Stride); }
    //     DEVICE constexpr Tensor3D _d0(int d0) const { return Tensor3D(get() + d0 * _D0_Stride); }
    // };

    // /// A base class for a 4-dimensional index.
    // template <typename _data_type,
    //           int _D1, int _D2, int _D3,
    //           int _D3_Stride = 1, int _D2_Stride = _D3 * _D3_Stride, int _D1_Stride = _D2 * _D2_Stride, int _D0_Stride = _D1 * _D1_Stride>
    // class Tensor4D : public TensorBase<_data_type>
    // {
    // public:
    //     using TensorBase<_data_type>::TensorBase;
    //     using TensorBase<_data_type>::get;

    //     DEVICE constexpr Tensor4D _d3(int d3) const { return Tensor4D(get() + d3 * _D3_Stride); }
    //     DEVICE constexpr Tensor4D _d2(int d2) const { return Tensor4D(get() + d2 * _D2_Stride); }
    //     DEVICE constexpr Tensor4D _d1(int d1) const { return Tensor4D(get() + d1 * _D1_Stride); }
    //     DEVICE constexpr Tensor4D _d0(int d0) const { return Tensor4D(get() + d0 * _D0_Stride); }
    // };

    // /// A base class for a 5-dimensional index.
    // template <typename _data_type,
    //           int _D1, int _D2, int _D3, int _D4,
    //           int _D4_Stride = 1, int _D3_Stride = _D4 * _D4_Stride, int _D2_Stride = _D3 * _D3_Stride,
    //           int _D1_Stride = _D2 * _D2_Stride, int _D0_Stride = _D1 * _D1_Stride>
    // class Tensor5D : public TensorBase<_data_type>
    // {
    // public:
    //     using TensorBase<_data_type>::TensorBase;
    //     using TensorBase<_data_type>::get;

    //     DEVICE constexpr Tensor5D _d4(int d4) const { return Tensor5D(get() + d4 * _D4_Stride); }
    //     DEVICE constexpr Tensor5D _d3(int d3) const { return Tensor5D(get() + d3 * _D3_Stride); }
    //     DEVICE constexpr Tensor5D _d2(int d2) const { return Tensor5D(get() + d2 * _D2_Stride); }
    //     DEVICE constexpr Tensor5D _d1(int d1) const { return Tensor5D(get() + d1 * _D1_Stride); }
    //     DEVICE constexpr Tensor5D _d0(int d0) const { return Tensor5D(get() + d0 * _D0_Stride); }
    // };

    // /// A base class for a 6-dimensional index.
    // template <typename _data_type,
    //           int _D1, int _D2, int _D3, int _D4, int _D5,
    //           int _D5_Stride = 1, int _D4_Stride = _D5 * _D5_Stride, int _D3_Stride = _D4 * _D4_Stride,
    //           int _D2_Stride = _D3 * _D3_Stride, int _D1_Stride = _D2 * _D2_Stride, int _D0_Stride = _D1 * _D1_Stride>
    // class Tensor6D : public TensorBase<_data_type>
    // {
    // public:
    //     using TensorBase<_data_type>::TensorBase;
    //     using TensorBase<_data_type>::get;

    //     DEVICE constexpr Tensor6D _d5(int d5) const { return Tensor6D(get() + d5 * _D5_Stride); }
    //     DEVICE constexpr Tensor6D _d4(int d4) const { return Tensor6D(get() + d4 * _D4_Stride); }
    //     DEVICE constexpr Tensor6D _d3(int d3) const { return Tensor6D(get() + d3 * _D3_Stride); }
    //     DEVICE constexpr Tensor6D _d2(int d2) const { return Tensor6D(get() + d2 * _D2_Stride); }
    //     DEVICE constexpr Tensor6D _d1(int d1) const { return Tensor6D(get() + d1 * _D1_Stride); }
    //     DEVICE constexpr Tensor6D _d0(int d0) const { return Tensor6D(get() + d0 * _D0_Stride); }
    // };

    // /// A base class for a 7-dimensional index.
    // template <typename _data_type,
    //           int _D1, int _D2, int _D3, int _D4, int _D5, int _D6,
    //           int _D6_Stride = 1, int _D5_Stride = _D6 * _D6_Stride, int _D4_Stride = _D5 * _D5_Stride,
    //           int _D3_Stride = _D4 * _D4_Stride, int _D2_Stride = _D3 * _D3_Stride, int _D1_Stride = _D2 * _D2_Stride,
    //           int _D0_Stride = _D1 * _D1_Stride>
    // class Tensor7D : public TensorBase<_data_type>
    // {
    // public:
    //     using TensorBase<_data_type>::TensorBase;
    //     using TensorBase<_data_type>::get;

    //     DEVICE constexpr Tensor7D _d6(int d6) const { return Tensor7D(get() + d6 * _D6_Stride); }
    //     DEVICE constexpr Tensor7D _d5(int d5) const { return Tensor7D(get() + d5 * _D5_Stride); }
    //     DEVICE constexpr Tensor7D _d4(int d4) const { return Tensor7D(get() + d4 * _D4_Stride); }
    //     DEVICE constexpr Tensor7D _d3(int d3) const { return Tensor7D(get() + d3 * _D3_Stride); }
    //     DEVICE constexpr Tensor7D _d2(int d2) const { return Tensor7D(get() + d2 * _D2_Stride); }
    //     DEVICE constexpr Tensor7D _d1(int d1) const { return Tensor7D(get() + d1 * _D1_Stride); }
    //     DEVICE constexpr Tensor7D _d0(int d0) const { return Tensor7D(get() + d0 * _D0_Stride); }
    // };
}

#endif
