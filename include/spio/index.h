#ifndef SPIO_INDEX_H_
#define SPIO_INDEX_H_

namespace spio
{
    /// A base class for a 2-dimensional index.
    template <int _D1>
    class Index2D
    {
    public:
        constexpr static int D1 = _D1;
        constexpr static int D0_Stride = D1;

        constexpr Index2D(int offset = 0) : _offset(offset) {}

        constexpr operator int() const { return _offset; }

        constexpr Index2D _d1(int d1) { return Index2D(_offset + d1); }
        constexpr Index2D _d0(int d0) { return Index2D(_offset + d0 * D0_Stride); }

        constexpr int _d0() { return _offset / D0_Stride; }
        constexpr int _d1() { return _offset % D1; }

    private:
        const int _offset;
    };

    /// A base class for a 3-dimensional index.
    template <int _D1, int _D2>
    class Index3D
    {
    public:
        constexpr static int D1 = _D1;
        constexpr static int D2 = _D2;

        constexpr static int D1_Stride = D2;
        constexpr static int D0_Stride = D1 * D2;

        constexpr Index3D(int offset = 0) : _offset(offset) {}

        constexpr operator int() const { return _offset; }

        constexpr Index3D _d2(int d2) { return Index3D(_offset + d2); }
        constexpr Index3D _d1(int d1) { return Index3D(_offset + d1 * D1_Stride); }
        constexpr Index3D _d0(int d0) { return Index3D(_offset + d0 * D0_Stride); }

        constexpr int _d2() { return _offset % D2; }
        constexpr int _d1() { return (_offset / D1_Stride) % D1; }
        constexpr int _d0() { return _offset / D0_Stride; }

    private:
        const int _offset;
    };

    /// A base class for a 4-dimensional index.
    template <int _D1, int _D2, int _D3>
    class Index4D
    {
    public:
        constexpr static int _D2_Stride = _D3;
        constexpr static int _D1_Stride = _D2 * _D3;
        constexpr static int _D0_Stride = _D1 * _D2 * _D3;

        constexpr Index4D(int offset = 0) : _offset(offset) {}

        constexpr operator int() const { return _offset; }

        constexpr Index4D _d3(int d3) { return Index4D(_offset + d3); }
        constexpr Index4D _d2(int d2) { return Index4D(_offset + d2 * _D2_Stride); }
        constexpr Index4D _d1(int d1) { return Index4D(_offset + d1 * _D1_Stride); }
        constexpr Index4D _d0(int d0) { return Index4D(_offset + d0 * _D0_Stride); }

        constexpr int _d3() { return _offset % _D3; }
        constexpr int _d2() { return (_offset / _D2_Stride) % _D2; }
        constexpr int _d1() { return (_offset / _D1_Stride) % _D1; }
        constexpr int _d0() { return _offset / _D0_Stride; }

    private:
        const int _offset;
    };
}
#endif
