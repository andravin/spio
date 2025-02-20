#ifndef SPIO_INDEX_H_
#define SPIO_INDEX_H_

#include "spio/macros.h"
namespace spio
{
    /// A base class for all Index* classes.
    class IndexBase
    {
    public:
        DEVICE constexpr IndexBase(unsigned offset = 0) : _offset(offset) {}
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

        DEVICE constexpr int _d0() const { return offset() / _D0_Stride; }
        DEVICE constexpr int _d1() const { return (offset() / _D1_Stride) % _D1; }
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

        DEVICE constexpr int _d2() const { return (offset() / _D2_Stride) % _D2; }
        DEVICE constexpr int _d1() const { return (offset() / _D1_Stride) % _D1; }
        DEVICE constexpr int _d0() const { return offset() / _D0_Stride; }
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

        DEVICE constexpr int _d3() const { return (offset() / _D3_Stride) % _D3; }
        DEVICE constexpr int _d2() const { return (offset() / _D2_Stride) % _D2; }
        DEVICE constexpr int _d1() const { return (offset() / _D1_Stride) % _D1; }
        DEVICE constexpr int _d0() const { return offset() / _D0_Stride; }
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

        DEVICE constexpr int _d4() const { return (offset() / _D4_Stride) % _D4; }
        DEVICE constexpr int _d3() const { return (offset() / _D3_Stride) % _D3; }
        DEVICE constexpr int _d2() const { return (offset() / _D2_Stride) % _D2; }
        DEVICE constexpr int _d1() const { return (offset() / _D1_Stride) % _D1; }
        DEVICE constexpr int _d0() const { return offset() / _D0_Stride; }
    };

    /// A base class for a 6-dimensional index.
    template<unsigned _D1, unsigned _D2, unsigned _D3, unsigned _D4, unsigned _D5>
    class Index6D : public IndexBase
    {
    public:
        constexpr static unsigned _D5_Stride = 1;
        constexpr static unsigned _D4_Stride = _D5 * _D5_Stride;
        constexpr static unsigned _D3_Stride = _D4 * _D4_Stride;
        constexpr static unsigned _D2_Stride = _D3 * _D3_Stride;
        constexpr static unsigned _D1_Stride = _D2 * _D2_Stride;
        constexpr static unsigned _D0_Stride = _D1 * _D1_Stride;

        using IndexBase::IndexBase;

        DEVICE constexpr int _d5() const { return (offset() / _D5_Stride) % _D5; }
        DEVICE constexpr int _d4() const { return (offset() / _D4_Stride) % _D4; }
        DEVICE constexpr int _d3() const { return (offset() / _D3_Stride) % _D3; }
        DEVICE constexpr int _d2() const { return (offset() / _D2_Stride) % _D2; }
        DEVICE constexpr int _d1() const { return (offset() / _D1_Stride) % _D1; }
        DEVICE constexpr int _d0() const { return offset() / _D0_Stride; }
    };

    /// A base class for a 7-dimensional index.
    template<unsigned _D1, unsigned _D2, unsigned _D3, unsigned _D4, unsigned _D5, unsigned _D6>
    class Index7D : public IndexBase
    {
        public:
        constexpr static unsigned _D6_Stride = 1;
        constexpr static unsigned _D5_Stride = _D6 * _D6_Stride;
        constexpr static unsigned _D4_Stride = _D5 * _D5_Stride;
        constexpr static unsigned _D3_Stride = _D4 * _D4_Stride;
        constexpr static unsigned _D2_Stride = _D3 * _D3_Stride;
        constexpr static unsigned _D1_Stride = _D2 * _D2_Stride;
        constexpr static unsigned _D0_Stride = _D1 * _D1_Stride;

        using IndexBase::IndexBase;

        DEVICE constexpr int _d6() const { return (offset() / _D6_Stride) % _D6; }
        DEVICE constexpr int _d5() const { return (offset() / _D5_Stride) % _D5; }
        DEVICE constexpr int _d4() const { return (offset() / _D4_Stride) % _D4; }
        DEVICE constexpr int _d3() const { return (offset() / _D3_Stride) % _D3; }
        DEVICE constexpr int _d2() const { return (offset() / _D2_Stride) % _D2; }
        DEVICE constexpr int _d1() const { return (offset() / _D1_Stride) % _D1; }
        DEVICE constexpr int _d0() const { return offset() / _D0_Stride; }
    };
}

#endif
