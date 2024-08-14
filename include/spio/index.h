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
        DEVICE constexpr IndexBase(unsigned offset = 0) : _offset(offset) {}
        DEVICE constexpr operator unsigned() const { return _offset; }
        DEVICE constexpr unsigned offset() const { return _offset; }

    private:
        const unsigned _offset;
    };

    /// A base class for a 2-dimensional index.
    template <unsigned _D1>
    class Index2D : public IndexBase
    {
    public:
        constexpr static unsigned _D1_Stride = 1;
        constexpr static unsigned _D0_Stride = _D1 * _D1_Stride;

        using IndexBase::IndexBase;

        DEVICE constexpr Index2D _d1(unsigned d1) const { return Index2D(offset() + d1 * _D1_Stride); }
        DEVICE constexpr Index2D _d0(unsigned d0) const { return Index2D(offset() + d0 * _D0_Stride); }

        DEVICE constexpr unsigned _d0() const { return offset() / _D0_Stride; }
        DEVICE constexpr unsigned _d1() const { return (offset() / _D1_Stride) % _D1; }
    };

    /// A base class for a 3-dimensional index.
    template <unsigned _D1, unsigned _D2>
    class Index3D : public IndexBase
    {
    public:
        constexpr static unsigned _D2_Stride = 1;
        constexpr static unsigned _D1_Stride = _D2 * _D2_Stride;
        constexpr static unsigned _D0_Stride = _D1 * _D1_Stride;

        using IndexBase::IndexBase;

        DEVICE constexpr Index3D _d2(unsigned d2) const { return Index3D(offset() + d2 * _D2_Stride); }
        DEVICE constexpr Index3D _d1(unsigned d1) const { return Index3D(offset() + d1 * _D1_Stride); }
        DEVICE constexpr Index3D _d0(unsigned d0) const { return Index3D(offset() + d0 * _D0_Stride); }

        DEVICE constexpr unsigned _d2() const { return (offset() / _D2_Stride) % _D2; }
        DEVICE constexpr unsigned _d1() const { return (offset() / _D1_Stride) % _D1; }
        DEVICE constexpr unsigned _d0() const { return offset() / _D0_Stride; }
    };

    /// A base class for a 4-dimensional index.
    template <unsigned _D1, unsigned _D2, unsigned _D3>
    class Index4D : public IndexBase
    {
    public:
        constexpr static unsigned _D3_Stride = 1;
        constexpr static unsigned _D2_Stride = _D3 * _D3_Stride;
        constexpr static unsigned _D1_Stride = _D2 * _D2_Stride;
        constexpr static unsigned _D0_Stride = _D1 * _D1_Stride;

        using IndexBase::IndexBase;

        DEVICE constexpr Index4D _d3(unsigned d3) const { return Index4D(offset() + d3 * _D3_Stride); }
        DEVICE constexpr Index4D _d2(unsigned d2) const { return Index4D(offset() + d2 * _D2_Stride); }
        DEVICE constexpr Index4D _d1(unsigned d1) const { return Index4D(offset() + d1 * _D1_Stride); }
        DEVICE constexpr Index4D _d0(unsigned d0) const { return Index4D(offset() + d0 * _D0_Stride); }

        DEVICE constexpr unsigned _d3() const { return (offset() / _D3_Stride) % _D3; }
        DEVICE constexpr unsigned _d2() const { return (offset() / _D2_Stride) % _D2; }
        DEVICE constexpr unsigned _d1() const { return (offset() / _D1_Stride) % _D1; }
        DEVICE constexpr unsigned _d0() const { return offset() / _D0_Stride; }
    };

    /// A base class for a 5-dimensional index.
    template <unsigned _D1, unsigned _D2, unsigned _D3, unsigned _D4>
    class Index5D : public IndexBase
    {
    public:
        constexpr static unsigned _D4_Stride = 1;
        constexpr static unsigned _D3_Stride = _D4 * _D4_Stride;
        constexpr static unsigned _D2_Stride = _D3 * _D3_Stride;
        constexpr static unsigned _D1_Stride = _D2 * _D2_Stride;
        constexpr static unsigned _D0_Stride = _D1 * _D1_Stride;

        using IndexBase::IndexBase;

        DEVICE constexpr Index5D _d4(unsigned d4) const { return Index5D(offset() + d4 * _D4_Stride); }
        DEVICE constexpr Index5D _d3(unsigned d3) const { return Index5D(offset() + d3 * _D3_Stride); }
        DEVICE constexpr Index5D _d2(unsigned d2) const { return Index5D(offset() + d2 * _D2_Stride); }
        DEVICE constexpr Index5D _d1(unsigned d1) const { return Index5D(offset() + d1 * _D1_Stride); }
        DEVICE constexpr Index5D _d0(unsigned d0) const { return Index5D(offset() + d0 * _D0_Stride); }

        DEVICE constexpr unsigned _d4() const { return (offset() / _D4_Stride) % _D4; }
        DEVICE constexpr unsigned _d3() const { return (offset() / _D3_Stride) % _D3; }
        DEVICE constexpr unsigned _d2() const { return (offset() / _D2_Stride) % _D2; }
        DEVICE constexpr unsigned _d1() const { return (offset() / _D1_Stride) % _D1; }
        DEVICE constexpr unsigned _d0() const { return offset() / _D0_Stride; }
    };
}

#endif
