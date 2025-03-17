#error This file is deprecated. Use tensor_variadic.h instead.

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

    template <typename _data_type, class _dim_0, int _stride_0>
    class Cursor1D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;
        using dim_0 = _dim_0;
        static constexpr int stride_0 = _stride_0;

        DEVICE constexpr Cursor1D operator[](dim_0 d0) const { return Cursor1D(get() + d0.get() * stride_0); }
    };

    template <typename _data_type, class _dim_0, class _dim_1, int _stride_1, int _stride_0>
    class Cursor2D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;
        using dim_0 = _dim_0;
        using dim_1 = _dim_1;

        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;

        DEVICE constexpr Cursor2D operator[](dim_0 d0) const { return Cursor2D(get() + d0.get() * stride_0); }
        DEVICE constexpr Cursor2D operator[](dim_1 d1) const { return Cursor2D(get() + d1.get() * stride_1); }
    };

    template <typename _data_type, class _dim_0, class _dim_1, class _dim_2, int _stride_2, int _stride_1, int _stride_0>
    class Cursor3D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;
        using dim_0 = _dim_0;
        using dim_1 = _dim_1;
        using dim_2 = _dim_2;

        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;
        static constexpr int stride_2 = _stride_2;

        DEVICE constexpr Cursor3D operator[](dim_0 d0) const { return Cursor3D(get() + d0.get() * stride_0); }
        DEVICE constexpr Cursor3D operator[](dim_1 d1) const { return Cursor3D(get() + d1.get() * stride_1); }
        DEVICE constexpr Cursor3D operator[](dim_2 d2) const { return Cursor3D(get() + d2.get() * stride_2); }
    };

    template <typename _data_type, class _dim_0, class _dim_1, class _dim_2, class _dim_3, int _stride_3, int _stride_2, int _stride_1, int _stride_0>
    class Cursor4D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;
        using dim_0 = _dim_0;
        using dim_1 = _dim_1;
        using dim_2 = _dim_2;
        using dim_3 = _dim_3;

        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;
        static constexpr int stride_2 = _stride_2;
        static constexpr int stride_3 = _stride_3;

        DEVICE constexpr Cursor4D operator[](dim_0 d0) const { return Cursor4D(get() + d0.get() * stride_0); }
        DEVICE constexpr Cursor4D operator[](dim_1 d1) const { return Cursor4D(get() + d1.get() * stride_1); }
        DEVICE constexpr Cursor4D operator[](dim_2 d2) const { return Cursor4D(get() + d2.get() * stride_2); }
        DEVICE constexpr Cursor4D operator[](dim_3 d3) const { return Cursor4D(get() + d3.get() * stride_3); }
    };

    template <typename _data_type, class _dim_0, class _dim_1, class _dim_2, class _dim_3, class _dim_4, int _stride_4, int _stride_3, int _stride_2, int _stride_1, int _stride_0>
    class Cursor5D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;
        using dim_0 = _dim_0;
        using dim_1 = _dim_1;
        using dim_2 = _dim_2;
        using dim_3 = _dim_3;
        using dim_4 = _dim_4;

        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;
        static constexpr int stride_2 = _stride_2;
        static constexpr int stride_3 = _stride_3;
        static constexpr int stride_4 = _stride_4;

        DEVICE constexpr Cursor5D operator[](dim_0 d0) const { return Cursor5D(get() + d0.get() * stride_0); }
        DEVICE constexpr Cursor5D operator[](dim_1 d1) const { return Cursor5D(get() + d1.get() * stride_1); }
        DEVICE constexpr Cursor5D operator[](dim_2 d2) const { return Cursor5D(get() + d2.get() * stride_2); }
        DEVICE constexpr Cursor5D operator[](dim_3 d3) const { return Cursor5D(get() + d3.get() * stride_3); }
        DEVICE constexpr Cursor5D operator[](dim_4 d4) const { return Cursor5D(get() + d4.get() * stride_4); }
    };

    template <typename _data_type, class _dim_0, class _dim_1, class _dim_2, class _dim_3, class _dim_4, class _dim_5, int _stride_5, int _stride_4, int _stride_3, int _stride_2, int _stride_1, int _stride_0>
    class Cursor6D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;
        using dim_0 = _dim_0;
        using dim_1 = _dim_1;
        using dim_2 = _dim_2;
        using dim_3 = _dim_3;
        using dim_4 = _dim_4;
        using dim_5 = _dim_5;

        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;
        static constexpr int stride_2 = _stride_2;
        static constexpr int stride_3 = _stride_3;
        static constexpr int stride_4 = _stride_4;
        static constexpr int stride_5 = _stride_5;

        DEVICE constexpr Cursor6D operator[](dim_0 d0) const { return Cursor6D(get() + d0.get() * stride_0); }
        DEVICE constexpr Cursor6D operator[](dim_1 d1) const { return Cursor6D(get() + d1.get() * stride_1); }
        DEVICE constexpr Cursor6D operator[](dim_2 d2) const { return Cursor6D(get() + d2.get() * stride_2); }
        DEVICE constexpr Cursor6D operator[](dim_3 d3) const { return Cursor6D(get() + d3.get() * stride_3); }
        DEVICE constexpr Cursor6D operator[](dim_4 d4) const { return Cursor6D(get() + d4.get() * stride_4); }
        DEVICE constexpr Cursor6D operator[](dim_5 d5) const { return Cursor6D(get() + d5.get() * stride_5); }
    };

    template <typename _data_type, class _dim_0, class _dim_1, class _dim_2, class _dim_3, class _dim_4, class _dim_5, class _dim_6, int _stride_6, int _stride_5, int _stride_4, int _stride_3, int _stride_2, int _stride_1, int _stride_0>
    class Cursor7D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;
        using dim_0 = _dim_0;
        using dim_1 = _dim_1;
        using dim_2 = _dim_2;
        using dim_3 = _dim_3;
        using dim_4 = _dim_4;
        using dim_5 = _dim_5;
        using dim_6 = _dim_6;

        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;
        static constexpr int stride_2 = _stride_2;
        static constexpr int stride_3 = _stride_3;
        static constexpr int stride_4 = _stride_4;
        static constexpr int stride_5 = _stride_5;
        static constexpr int stride_6 = _stride_6;

        DEVICE constexpr Cursor7D operator[](dim_0 d0) const { return Cursor7D(get() + d0.get() * stride_0); }
        DEVICE constexpr Cursor7D operator[](dim_1 d1) const { return Cursor7D(get() + d1.get() * stride_1); }
        DEVICE constexpr Cursor7D operator[](dim_2 d2) const { return Cursor7D(get() + d2.get() * stride_2); }
        DEVICE constexpr Cursor7D operator[](dim_3 d3) const { return Cursor7D(get() + d3.get() * stride_3); }
        DEVICE constexpr Cursor7D operator[](dim_4 d4) const { return Cursor7D(get() + d4.get() * stride_4); }
        DEVICE constexpr Cursor7D operator[](dim_5 d5) const { return Cursor7D(get() + d5.get() * stride_5); }
        DEVICE constexpr Cursor7D operator[](dim_6 d6) const { return Cursor7D(get() + d6.get() * stride_6); }
    };

    template <typename _data_type, class _dim_0, int _size_0, int _stride_0 = 1>
    class Tensor1D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;

        using dim_0 = _dim_0;

        using cursor_type = Cursor1D<data_type, _dim_0, _stride_0>;

        static constexpr dim_0 size_0 = _size_0;
        static constexpr int stride_0 = _stride_0;
        static constexpr int size = size_0.get();

        DEVICE constexpr cursor_type operator[](dim_0 d0) const { return cursor_type(get())[d0]; }

        template <int _size>
        DEVICE constexpr auto slice(dim_0 slice_start)
        {
            return Tensor1D<data_type, dim_0, _size, stride_0>((*this)[slice_start].get());
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

        using cursor_type = Cursor2D<data_type, _dim_0, _dim_1, _stride_1, _stride_0>;

        static constexpr dim_0 size_0 = _size_0;
        static constexpr dim_1 size_1 = _size_1;
        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;
        static constexpr int size = size_0.get() * size_1.get();

        DEVICE constexpr cursor_type operator[](dim_0 d0) const { return cursor_type(get())[d0]; }
        DEVICE constexpr cursor_type operator[](dim_1 d1) const { return cursor_type(get())[d1]; }

        template <int _size>
        DEVICE constexpr auto slice(dim_0 slice_start)
        {
            return Tensor2D<data_type, dim_0, dim_1, _size, size_1.get(), stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_1 slice_start)
        {
            return Tensor2D<data_type, dim_0, dim_1, size_0.get(), _size, stride_1, stride_0>((*this)[slice_start].get());
        }
    };

    template <
        typename _data_type,
        class _dim_0,
        class _dim_1,
        class _dim_2,
        int _size_0,
        int _size_1,
        int _size_2,
        int _stride_2 = 1,
        int _stride_1 = _size_2 * _stride_2,
        int _stride_0 = _size_1 * _stride_1>
    class Tensor3D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;

        using dim_0 = _dim_0;
        using dim_1 = _dim_1;
        using dim_2 = _dim_2;

        using cursor_type = Cursor3D<data_type, _dim_0, _dim_1, _dim_2, _stride_2, _stride_1, _stride_0>;

        static constexpr dim_0 size_0 = _size_0;
        static constexpr dim_1 size_1 = _size_1;
        static constexpr dim_2 size_2 = _size_2;
        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;
        static constexpr int stride_2 = _stride_2;
        static constexpr int size = size_0.get() * size_1.get() * size_2.get();

        DEVICE constexpr cursor_type operator[](dim_0 d0) const { return cursor_type(get())[d0]; }
        DEVICE constexpr cursor_type operator[](dim_1 d1) const { return cursor_type(get())[d1]; }
        DEVICE constexpr cursor_type operator[](dim_2 d2) const { return cursor_type(get())[d2]; }

        template <int _size>
        DEVICE constexpr auto slice(dim_0 slice_start)
        {
            return Tensor3D<data_type, dim_0, dim_1, dim_2, _size, size_1.get(), size_2.get(), stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_1 slice_start)
        {
            return Tensor3D<data_type, dim_0, dim_1, dim_2, size_0.get(), _size, size_2.get(), stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_2 slice_start)
        {
            return Tensor3D<data_type, dim_0, dim_1, dim_2, size_0.get(), size_1.get(), _size, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }
    };

    template <
        typename _data_type,
        class _dim_0,
        class _dim_1,
        class _dim_2,
        class _dim_3,
        int _size_0,
        int _size_1,
        int _size_2,
        int _size_3,
        int _stride_3 = 1,
        int _stride_2 = _size_3 * _stride_3,
        int _stride_1 = _size_2 * _stride_2,
        int _stride_0 = _size_1 * _stride_1>
    class Tensor4D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;

        using dim_0 = _dim_0;
        using dim_1 = _dim_1;
        using dim_2 = _dim_2;
        using dim_3 = _dim_3;

        using cursor_type = Cursor4D<data_type, _dim_0, _dim_1, _dim_2, _dim_3, _stride_3, _stride_2, _stride_1, _stride_0>;

        static constexpr dim_0 size_0 = _size_0;
        static constexpr dim_1 size_1 = _size_1;
        static constexpr dim_2 size_2 = _size_2;
        static constexpr dim_3 size_3 = _size_3;
        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;
        static constexpr int stride_2 = _stride_2;
        static constexpr int stride_3 = _stride_3;
        static constexpr int size = size_0.get() * size_1.get() * size_2.get() * size_3.get();

        DEVICE constexpr cursor_type operator[](dim_0 d0) const { return cursor_type(get())[d0]; }
        DEVICE constexpr cursor_type operator[](dim_1 d1) const { return cursor_type(get())[d1]; }
        DEVICE constexpr cursor_type operator[](dim_2 d2) const { return cursor_type(get())[d2]; }
        DEVICE constexpr cursor_type operator[](dim_3 d3) const { return cursor_type(get())[d3]; }

        template <int _size>
        DEVICE constexpr auto slice(dim_0 slice_start)
        {
            return Tensor4D<data_type, dim_0, dim_1, dim_2, dim_3, _size, size_1.get(), size_2.get(), size_3.get(), stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_1 slice_start)
        {
            return Tensor4D<data_type, dim_0, dim_1, dim_2, dim_3, size_0.get(), _size, size_2.get(), size_3.get(), stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }
        template <int _size>
        DEVICE constexpr auto slice(dim_2 slice_start)
        {
            return Tensor4D<data_type, dim_0, dim_1, dim_2, dim_3, size_0.get(), size_1.get(), _size, size_3.get(), stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_3 slice_start)
        {
            return Tensor4D<data_type, dim_0, dim_1, dim_2, dim_3, size_0.get(), size_1.get(), size_2.get(), _size, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }
    };

    template <
        typename _data_type,
        class _dim_0,
        class _dim_1,
        class _dim_2,
        class _dim_3,
        class _dim_4,
        int _size_0,
        int _size_1,
        int _size_2,
        int _size_3,
        int _size_4,
        int _stride_4 = 1,
        int _stride_3 = _size_4 * _stride_4,
        int _stride_2 = _size_3 * _stride_3,
        int _stride_1 = _size_2 * _stride_2,
        int _stride_0 = _size_1 * _stride_1>
    class Tensor5D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;

        using dim_0 = _dim_0;
        using dim_1 = _dim_1;
        using dim_2 = _dim_2;
        using dim_3 = _dim_3;
        using dim_4 = _dim_4;

        using cursor_type = Cursor5D<data_type, _dim_0, _dim_1, _dim_2, _dim_3, _dim_4, _stride_4, _stride_3, _stride_2, _stride_1, _stride_0>;

        static constexpr dim_0 size_0 = _size_0;
        static constexpr dim_1 size_1 = _size_1;
        static constexpr dim_2 size_2 = _size_2;
        static constexpr dim_3 size_3 = _size_3;
        static constexpr dim_4 size_4 = _size_4;
        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;
        static constexpr int stride_2 = _stride_2;
        static constexpr int stride_3 = _stride_3;
        static constexpr int stride_4 = _stride_4;
        static constexpr int size = size_0.get() * size_1.get() * size_2.get() * size_3.get() * size_4.get();

        DEVICE constexpr cursor_type operator[](dim_0 d0) const { return cursor_type(get())[d0]; }
        DEVICE constexpr cursor_type operator[](dim_1 d1) const { return cursor_type(get())[d1]; }
        DEVICE constexpr cursor_type operator[](dim_2 d2) const { return cursor_type(get())[d2]; }
        DEVICE constexpr cursor_type operator[](dim_3 d3) const { return cursor_type(get())[d3]; }
        DEVICE constexpr cursor_type operator[](dim_4 d4) const { return cursor_type(get())[d4]; }

        template <int _size>
        DEVICE constexpr auto slice(dim_0 slice_start)
        {
            return Tensor5D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, _size, size_1.get(), size_2.get(), size_3.get(), size_4.get(), stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_1 slice_start)
        {
            return Tensor5D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, size_0.get(), _size, size_2.get(), size_3.get(), size_4.get(), stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_2 slice_start)
        {
            return Tensor5D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, size_0.get(), size_1.get(), _size, size_3.get(), size_4.get(), stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_3 slice_start)
        {
            return Tensor5D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, size_0.get(), size_1.get(), size_2.get(), _size, size_4.get(), stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_4 slice_start)
        {
            return Tensor5D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, size_0.get(), size_1.get(), size_2.get(), size_3.get(), _size, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }
    };

    template <
        typename _data_type,
        class _dim_0,
        class _dim_1,
        class _dim_2,
        class _dim_3,
        class _dim_4,
        class _dim_5,
        int _size_0,
        int _size_1,
        int _size_2,
        int _size_3,
        int _size_4,
        int _size_5,
        int _stride_5 = 1,
        int _stride_4 = _size_5 * _stride_5,
        int _stride_3 = _size_4 * _stride_4,
        int _stride_2 = _size_3 * _stride_3,
        int _stride_1 = _size_2 * _stride_2,
        int _stride_0 = _size_1 * _stride_1>
    class Tensor6D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;

        using dim_0 = _dim_0;
        using dim_1 = _dim_1;
        using dim_2 = _dim_2;
        using dim_3 = _dim_3;
        using dim_4 = _dim_4;
        using dim_5 = _dim_5;

        using cursor_type = Cursor6D<data_type, _dim_0, _dim_1, _dim_2, _dim_3, _dim_4, _dim_5, _stride_5, _stride_4, _stride_3, _stride_2, _stride_1, _stride_0>;

        static constexpr dim_0 size_0 = _size_0;
        static constexpr dim_1 size_1 = _size_1;
        static constexpr dim_2 size_2 = _size_2;
        static constexpr dim_3 size_3 = _size_3;
        static constexpr dim_4 size_4 = _size_4;
        static constexpr dim_5 size_5 = _size_5;
        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;
        static constexpr int stride_2 = _stride_2;
        static constexpr int stride_3 = _stride_3;
        static constexpr int stride_4 = _stride_4;
        static constexpr int stride_5 = _stride_5;
        static constexpr int size = size_0.get() * size_1.get() * size_2.get() * size_3.get() * size_4.get() * size_5.get();

        DEVICE constexpr cursor_type operator[](dim_0 d0) const { return cursor_type(get())[d0]; }
        DEVICE constexpr cursor_type operator[](dim_1 d1) const { return cursor_type(get())[d1]; }
        DEVICE constexpr cursor_type operator[](dim_2 d2) const { return cursor_type(get())[d2]; }
        DEVICE constexpr cursor_type operator[](dim_3 d3) const { return cursor_type(get())[d3]; }
        DEVICE constexpr cursor_type operator[](dim_4 d4) const { return cursor_type(get())[d4]; }
        DEVICE constexpr cursor_type operator[](dim_5 d5) const { return cursor_type(get())[d5]; }

        template <int _size>
        DEVICE constexpr auto slice(dim_0 slice_start)
        {
            return Tensor6D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, _size, size_1.get(), size_2.get(), size_3.get(), size_4.get(), size_5.get(), stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_1 slice_start)
        {
            return Tensor6D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, size_0.get(), _size, size_2.get(), size_3.get(), size_4.get(), size_5.get(), stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_2 slice_start)
        {
            return Tensor6D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, size_0.get(), size_1.get(), _size, size_3.get(), size_4.get(), size_5.get(), stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_3 slice_start)
        {
            return Tensor6D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, size_0.get(), size_1.get(), size_2.get(), _size, size_4.get(), size_5.get(), stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_4 slice_start)
        {
            return Tensor6D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, size_0.get(), size_1.get(), size_2.get(), size_3.get(), _size, size_5.get(), stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_5 slice_start)
        {
            return Tensor6D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, size_0.get(), size_1.get(), size_2.get(), size_3.get(), size_4.get(), _size, stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }
    };

    template <
        typename _data_type,
        class _dim_0,
        class _dim_1,
        class _dim_2,
        class _dim_3,
        class _dim_4,
        class _dim_5,
        class _dim_6,
        int _size_0,
        int _size_1,
        int _size_2,
        int _size_3,
        int _size_4,
        int _size_5,
        int _size_6,
        int _stride_6 = 1,
        int _stride_5 = _size_6 * _stride_6,
        int _stride_4 = _size_5 * _stride_5,
        int _stride_3 = _size_4 * _stride_4,
        int _stride_2 = _size_3 * _stride_3,
        int _stride_1 = _size_2 * _stride_2,
        int _stride_0 = _size_1 * _stride_1>
    class Tensor7D : public TensorBase<_data_type>
    {
    public:
        using data_type = _data_type;
        using TensorBase<data_type>::TensorBase;
        using TensorBase<data_type>::get;

        using dim_0 = _dim_0;
        using dim_1 = _dim_1;
        using dim_2 = _dim_2;
        using dim_3 = _dim_3;
        using dim_4 = _dim_4;
        using dim_5 = _dim_5;
        using dim_6 = _dim_6;

        using cursor_type = Cursor7D<data_type, _dim_0, _dim_1, _dim_2, _dim_3, _dim_4, _dim_5, _dim_6, _stride_6, _stride_5, _stride_4, _stride_3, _stride_2, _stride_1, _stride_0>;

        static constexpr dim_0 size_0 = _size_0;
        static constexpr dim_1 size_1 = _size_1;
        static constexpr dim_2 size_2 = _size_2;
        static constexpr dim_3 size_3 = _size_3;
        static constexpr dim_4 size_4 = _size_4;
        static constexpr dim_5 size_5 = _size_5;
        static constexpr dim_6 size_6 = _size_6;
        static constexpr int stride_0 = _stride_0;
        static constexpr int stride_1 = _stride_1;
        static constexpr int stride_2 = _stride_2;
        static constexpr int stride_3 = _stride_3;
        static constexpr int stride_4 = _stride_4;
        static constexpr int stride_5 = _stride_5;
        static constexpr int stride_6 = _stride_6;
        static constexpr int size = size_0.get() * size_1.get() * size_2.get() * size_3.get() * size_4.get() * size_5.get() * size_6.get();

        DEVICE constexpr cursor_type operator[](dim_0 d0) const { return cursor_type(get())[d0]; }
        DEVICE constexpr cursor_type operator[](dim_1 d1) const { return cursor_type(get())[d1]; }
        DEVICE constexpr cursor_type operator[](dim_2 d2) const { return cursor_type(get())[d2]; }
        DEVICE constexpr cursor_type operator[](dim_3 d3) const { return cursor_type(get())[d3]; }
        DEVICE constexpr cursor_type operator[](dim_4 d4) const { return cursor_type(get())[d4]; }
        DEVICE constexpr cursor_type operator[](dim_5 d5) const { return cursor_type(get())[d5]; }
        DEVICE constexpr cursor_type operator[](dim_6 d6) const { return cursor_type(get())[d6]; }

        template <int _size>
        DEVICE constexpr auto slice(dim_0 slice_start)
        {
            return Tensor7D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, _size, size_1.get(), size_2.get(), size_3.get(), size_4.get(), size_5.get(), size_6.get(), stride_6, stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_1 slice_start)
        {
            return Tensor7D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, size_0.get(), _size, size_2.get(), size_3.get(), size_4.get(), size_5.get(), size_6.get(), stride_6, stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_2 slice_start)
        {
            return Tensor7D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, size_0.get(), size_1.get(), _size, size_3.get(), size_4.get(), size_5.get(), size_6.get(), stride_6, stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_3 slice_start)
        {
            return Tensor7D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, size_0.get(), size_1.get(), size_2.get(), _size, size_4.get(), size_5.get(), size_6.get(), stride_6, stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_4 slice_start)
        {
            return Tensor7D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, size_0.get(), size_1.get(), size_2.get(), size_3.get(), _size, size_5.get(), size_6.get(), stride_6, stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_5 slice_start)
        {
            return Tensor7D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, size_0.get(), size_1.get(), size_2.get(), size_3.get(), size_4.get(), _size, size_6.get(), stride_6, stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }

        template <int _size>
        DEVICE constexpr auto slice(dim_6 slice_start)
        {
            return Tensor7D<data_type, dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, size_0.get(), size_1.get(), size_2.get(), size_3.get(), size_4.get(), size_5.get(), _size, stride_6, stride_5, stride_4, stride_3, stride_2, stride_1, stride_0>((*this)[slice_start].get());
        }
    };
}

#endif
