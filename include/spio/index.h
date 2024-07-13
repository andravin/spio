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
    /// A base class for all Index* classes.
    class IndexBase
    {
    public:
        DEVICE constexpr IndexBase(int offset = 0) : _offset(offset) {}
        DEVICE constexpr operator int() const { return _offset; }
        DEVICE constexpr int offset() const { return _offset; }

    private:
        const int _offset;
    };

    /// A base class for a 2-dimensional index.
    template <int _D1>
    class Index2D : public IndexBase
    {
    public:
        constexpr static int _D1_Stride = 1;
        constexpr static int _D0_Stride = _D1 * _D1_Stride;

        using IndexBase::IndexBase;

        DEVICE constexpr Index2D _d1(int d1) const { return Index2D(offset() + d1 * _D1_Stride); }
        DEVICE constexpr Index2D _d0(int d0) const { return Index2D(offset() + d0 * _D0_Stride); }

        DEVICE constexpr int _d0() const { return offset() / _D0_Stride; }
        DEVICE constexpr int _d1() const { return (offset() / _D1_Stride) % _D1; }
    };

    /// A base class for a 3-dimensional index.
    template <int _D1, int _D2>
    class Index3D : public IndexBase
    {
    public:
        constexpr static int _D2_Stride = 1;
        constexpr static int _D1_Stride = _D2 * _D2_Stride;
        constexpr static int _D0_Stride = _D1 * _D1_Stride;

        using IndexBase::IndexBase;

        DEVICE constexpr Index3D _d2(int d2) const { return Index3D(offset() + d2 * _D2_Stride); }
        DEVICE constexpr Index3D _d1(int d1) const { return Index3D(offset() + d1 * _D1_Stride); }
        DEVICE constexpr Index3D _d0(int d0) const { return Index3D(offset() + d0 * _D0_Stride); }

        DEVICE constexpr int _d2() const { return (offset() / _D2_Stride) % _D2; }
        DEVICE constexpr int _d1() const { return (offset() / _D1_Stride) % _D1; }
        DEVICE constexpr int _d0() const { return offset() / _D0_Stride; }
    };

    /// A base class for a 4-dimensional index.
    template <int _D1, int _D2, int _D3>
    class Index4D : public IndexBase
    {
    public:
        constexpr static int _D3_Stride = 1;
        constexpr static int _D2_Stride = _D3 * _D3_Stride;
        constexpr static int _D1_Stride = _D2 * _D2_Stride;
        constexpr static int _D0_Stride = _D1 * _D1_Stride;

        using IndexBase::IndexBase;

        DEVICE constexpr Index4D _d3(int d3) const { return Index4D(offset() + d3 * _D3_Stride); }
        DEVICE constexpr Index4D _d2(int d2) const { return Index4D(offset() + d2 * _D2_Stride); }
        DEVICE constexpr Index4D _d1(int d1) const { return Index4D(offset() + d1 * _D1_Stride); }
        DEVICE constexpr Index4D _d0(int d0) const { return Index4D(offset() + d0 * _D0_Stride); }

        DEVICE constexpr int _d3() const { return (offset() / _D3_Stride) % _D3; }
        DEVICE constexpr int _d2() const { return (offset() / _D2_Stride) % _D2; }
        DEVICE constexpr int _d1() const { return (offset() / _D1_Stride) % _D1; }
        DEVICE constexpr int _d0() const { return offset() / _D0_Stride; }
    };
}

#endif
