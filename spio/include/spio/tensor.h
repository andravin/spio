#ifndef SPIO_TENSOR_H_
#define SPIO_TENSOR_H_

#include "spio/macros.h"

namespace spio
{
    /// A base class for all Tensor* classes.
    template <typename DataType>
    class TensorBase
    {
    public:
        using data_type = DataType;
        static constexpr int element_size = sizeof(DataType);
        DEVICE constexpr TensorBase(DataType *data = nullptr) : _data(data) {}
        DEVICE constexpr DataType *get() const { return _data; }
        DEVICE void reset(DataType *data) { _data = data; }
        DEVICE constexpr DataType &operator*() const { return *_data; }
        DEVICE constexpr DataType *operator->() const { return _data; }

    private:
        DataType *_data;
    };

    template <typename DataType>
    class Tensor1D : public TensorBase<DataType>
    {
    public:
        using TensorBase<DataType>::TensorBase;
        using TensorBase<DataType>::get;

        DEVICE constexpr Tensor1D _d0(int d0) const { return Tensor1D(get() + d0); }
    };

    /// A base class for a 2-dimensional index.
    template <typename DataType, int _D1, int _D1_Stride = 1, int _D0_Stride = _D1 * _D1_Stride>
    class Tensor2D : public TensorBase<DataType>
    {
    public:
        using TensorBase<DataType>::TensorBase;
        using TensorBase<DataType>::get;

        DEVICE constexpr Tensor2D _d1(int d1) const { return Tensor2D(get() + d1 * _D1_Stride); }
        DEVICE constexpr Tensor2D _d0(int d0) const { return Tensor2D(get() + d0 * _D0_Stride); }
    };

    /// A base class for a 3-dimensional index.
    template <typename DataType,
              int _D1, int _D2,
              int _D2_Stride = 1, int _D1_Stride = _D2 * _D2_Stride, int _D0_Stride = _D1 * _D1_Stride>
    class Tensor3D : public TensorBase<DataType>
    {
    public:
        using TensorBase<DataType>::TensorBase;
        using TensorBase<DataType>::get;

        DEVICE constexpr Tensor3D _d2(int d2) const { return Tensor3D(get() + d2 * _D2_Stride); }
        DEVICE constexpr Tensor3D _d1(int d1) const { return Tensor3D(get() + d1 * _D1_Stride); }
        DEVICE constexpr Tensor3D _d0(int d0) const { return Tensor3D(get() + d0 * _D0_Stride); }
    };

    /// A base class for a 4-dimensional index.
    template <typename DataType,
              int _D1, int _D2, int _D3,
              int _D3_Stride = 1, int _D2_Stride = _D3 * _D3_Stride, int _D1_Stride = _D2 * _D2_Stride, int _D0_Stride = _D1 * _D1_Stride>
    class Tensor4D : public TensorBase<DataType>
    {
    public:
        using TensorBase<DataType>::TensorBase;
        using TensorBase<DataType>::get;

        DEVICE constexpr Tensor4D _d3(int d3) const { return Tensor4D(get() + d3 * _D3_Stride); }
        DEVICE constexpr Tensor4D _d2(int d2) const { return Tensor4D(get() + d2 * _D2_Stride); }
        DEVICE constexpr Tensor4D _d1(int d1) const { return Tensor4D(get() + d1 * _D1_Stride); }
        DEVICE constexpr Tensor4D _d0(int d0) const { return Tensor4D(get() + d0 * _D0_Stride); }
    };

    /// A base class for a 5-dimensional index.
    template <typename DataType,
              int _D1, int _D2, int _D3, int _D4,
              int _D4_Stride = 1, int _D3_Stride = _D4 * _D4_Stride, int _D2_Stride = _D3 * _D3_Stride,
              int _D1_Stride = _D2 * _D2_Stride, int _D0_Stride = _D1 * _D1_Stride>
    class Tensor5D : public TensorBase<DataType>
    {
    public:
        using TensorBase<DataType>::TensorBase;
        using TensorBase<DataType>::get;

        DEVICE constexpr Tensor5D _d4(int d4) const { return Tensor5D(get() + d4 * _D4_Stride); }
        DEVICE constexpr Tensor5D _d3(int d3) const { return Tensor5D(get() + d3 * _D3_Stride); }
        DEVICE constexpr Tensor5D _d2(int d2) const { return Tensor5D(get() + d2 * _D2_Stride); }
        DEVICE constexpr Tensor5D _d1(int d1) const { return Tensor5D(get() + d1 * _D1_Stride); }
        DEVICE constexpr Tensor5D _d0(int d0) const { return Tensor5D(get() + d0 * _D0_Stride); }
    };

    /// A base class for a 6-dimensional index.
    template <typename DataType,
              int _D1, int _D2, int _D3, int _D4, int _D5,
              int _D5_Stride = 1, int _D4_Stride = _D5 * _D5_Stride, int _D3_Stride = _D4 * _D4_Stride,
              int _D2_Stride = _D3 * _D3_Stride, int _D1_Stride = _D2 * _D2_Stride, int _D0_Stride = _D1 * _D1_Stride>
    class Tensor6D : public TensorBase<DataType>
    {
    public:
        using TensorBase<DataType>::TensorBase;
        using TensorBase<DataType>::get;

        DEVICE constexpr Tensor6D _d5(int d5) const { return Tensor6D(get() + d5 * _D5_Stride); }
        DEVICE constexpr Tensor6D _d4(int d4) const { return Tensor6D(get() + d4 * _D4_Stride); }
        DEVICE constexpr Tensor6D _d3(int d3) const { return Tensor6D(get() + d3 * _D3_Stride); }
        DEVICE constexpr Tensor6D _d2(int d2) const { return Tensor6D(get() + d2 * _D2_Stride); }
        DEVICE constexpr Tensor6D _d1(int d1) const { return Tensor6D(get() + d1 * _D1_Stride); }
        DEVICE constexpr Tensor6D _d0(int d0) const { return Tensor6D(get() + d0 * _D0_Stride); }
    };

    /// A base class for a 7-dimensional index.
    template <typename DataType,
              int _D1, int _D2, int _D3, int _D4, int _D5, int _D6,
              int _D6_Stride = 1, int _D5_Stride = _D6 * _D6_Stride, int _D4_Stride = _D5 * _D5_Stride,
              int _D3_Stride = _D4 * _D4_Stride, int _D2_Stride = _D3 * _D3_Stride, int _D1_Stride = _D2 * _D2_Stride,
              int _D0_Stride = _D1 * _D1_Stride>
    class Tensor7D : public TensorBase<DataType>
    {
    public:
        using TensorBase<DataType>::TensorBase;
        using TensorBase<DataType>::get;

        DEVICE constexpr Tensor7D _d6(int d6) const { return Tensor7D(get() + d6 * _D6_Stride); }
        DEVICE constexpr Tensor7D _d5(int d5) const { return Tensor7D(get() + d5 * _D5_Stride); }
        DEVICE constexpr Tensor7D _d4(int d4) const { return Tensor7D(get() + d4 * _D4_Stride); }
        DEVICE constexpr Tensor7D _d3(int d3) const { return Tensor7D(get() + d3 * _D3_Stride); }
        DEVICE constexpr Tensor7D _d2(int d2) const { return Tensor7D(get() + d2 * _D2_Stride); }
        DEVICE constexpr Tensor7D _d1(int d1) const { return Tensor7D(get() + d1 * _D1_Stride); }
        DEVICE constexpr Tensor7D _d0(int d0) const { return Tensor7D(get() + d0 * _D0_Stride); }
    };
}

#endif
