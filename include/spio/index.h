#ifndef SPIO_INDEX_H_
#define SPIO_INDEX_H_

namespace spio
{
    template <int _D1>
    class Index2D
    {
    public:
        constexpr static int D1 = _D1;

        constexpr Index2D(int offset = 0) : _offset(offset) {}

        constexpr operator int() const { return _offset; }

        constexpr Index2D _d1(int d1) { return Index2D(_offset + d1); }
        constexpr Index2D _d0(int d0) { return Index2D(_offset + d0 * _D1); }

    private:
        const int _offset;
    };
}

#endif
