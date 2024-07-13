#ifndef SPIO_INDEX_H_
#define SPIO_INDEX_H

#ifndef DEVICE
#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif
#endif
namespace spio
{
    /// A base class for all Tensor* classes.
    template<typename DataType>
    class TensorBase
    {
    public:
        DEVICE constexpr TensorBase(DataType *data = nullptr) : _data(data) {}
        DEVICE constexpr DataType *const get() const { return _data; }
        DEVICE constexpr DataType & operator*() const { return *_data; }
    private:
        DataType *const _data;
    };

    /// A base class for a 2-dimensional index.
    template <typename DataType, int _D1>
    class Tensor2D : public TensorBase<DataType>
    {
    public:
        constexpr static int _D1_Stride = 1;
        constexpr static int _D0_Stride = _D1 * _D1_Stride;

        using TensorBase<DataType>::TensorBase;
        using TensorBase<DataType>::get;

        DEVICE constexpr Tensor2D _d1(int d1) const { return Tensor2D(get() + d1 * _D1_Stride); }
        DEVICE constexpr Tensor2D _d0(int d0) const { return Tensor2D(get() + d0 * _D0_Stride); }
    };

    /// A base class for a 3-dimensional index.
    template <typename DataType, int _D1, int _D2>
    class Tensor3D : public TensorBase<DataType>
    {
    public:
        constexpr static int _D2_Stride = 1;
        constexpr static int _D1_Stride = _D2 * _D2_Stride;
        constexpr static int _D0_Stride = _D1 * _D1_Stride;

        using TensorBase<DataType>::TensorBase;
        using TensorBase<DataType>::get;

        DEVICE constexpr Tensor3D _d2(int d2) const { return Tensor3D(get() + d2 * _D2_Stride); }
        DEVICE constexpr Tensor3D _d1(int d1) const { return Tensor3D(get() + d1 * _D1_Stride); }
        DEVICE constexpr Tensor3D _d0(int d0) const { return Tensor3D(get() + d0 * _D0_Stride); }
    };

    /// A base class for a 4-dimensional index.
    template <typename DataType, int _D1, int _D2, int _D3>
    class Tensor4D : public TensorBase<DataType>
    {
    public:
        constexpr static int _D3_Stride = 1;
        constexpr static int _D2_Stride = _D3 * _D3_Stride;
        constexpr static int _D1_Stride = _D2 * _D2_Stride;
        constexpr static int _D0_Stride = _D1 * _D1_Stride;

        using TensorBase<DataType>::TensorBase;
        using TensorBase<DataType>::get;

        DEVICE constexpr Tensor4D _d3(int d3) const { return Tensor4D(get() + d3 * _D3_Stride); }
        DEVICE constexpr Tensor4D _d2(int d2) const { return Tensor4D(get() + d2 * _D2_Stride); }
        DEVICE constexpr Tensor4D _d1(int d1) const { return Tensor4D(get() + d1 * _D1_Stride); }
        DEVICE constexpr Tensor4D _d0(int d0) const { return Tensor4D(get() + d0 * _D0_Stride); }
    };
}

#endif
