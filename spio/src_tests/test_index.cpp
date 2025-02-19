#include "utest.h"
#include "spio/index.h"

UTEST(Index2D, index_from_offset)
{
    constexpr int D1 = 16;
    using Idx = spio::Index2D<D1>;
    EXPECT_EQ(Idx(53)._d0(), 53 / D1);
    EXPECT_EQ(Idx(53)._d1(), 53 % D1);
}

UTEST(Index3D, index_from_offset)
{
    constexpr int D1 = 23;
    constexpr int D2 = 47;
    using Idx = spio::Index3D<D1, D2>;
    constexpr int Offset = 2049;
    Idx idx(Offset);
    EXPECT_EQ(idx._d2(), Offset % D2);
    EXPECT_EQ(idx._d1(), (Offset / D2) % D1);
    EXPECT_EQ(idx._d0(), Offset / (D1 * D2));
}

UTEST(Index4D, index_from_offset)
{
    constexpr int D1 = 45;
    constexpr int D2 = 56;
    constexpr int D3 = 67;
    using Idx = spio::Index4D<D1, D2, D3>;
    constexpr int Offset = 867539;
    Idx idx(Offset);
    EXPECT_EQ(idx._d3(), Offset % D3);
    EXPECT_EQ(idx._d2(), (Offset / D3) % D2);
    EXPECT_EQ(idx._d1(), (Offset / (D2 * D3)) % D1);
    EXPECT_EQ(idx._d0(), Offset / (D1 * D2 * D3));
}

UTEST_MAIN()
