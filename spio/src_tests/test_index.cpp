#include "utest.h"
#include "spio/index.h"

UTEST(Index2D, offset_from_index)
{
    constexpr int D1 = 16;
    using Idx = spio::Index2D<D1>;
    EXPECT_EQ(static_cast<int>(Idx()._d1(10)), 10);
    EXPECT_EQ(static_cast<int>(Idx()._d0(9)._d1(10)), 9 * D1 + 10);
}

UTEST(Index2D, index_from_offset)
{
    constexpr int D1 = 16;
    using Idx = spio::Index2D<D1>;
    EXPECT_EQ(Idx(53)._d0(), 53 / D1);
    EXPECT_EQ(Idx(53)._d1(), 53 % D1);
}

UTEST(Index3D, offset_from_index)
{
    constexpr int D1 = 12;
    constexpr int D2 = 13;
    using Idx = spio::Index3D<D1, D2>;
    EXPECT_EQ(static_cast<int>(Idx()._d2(7)), 7);
    EXPECT_EQ(static_cast<int>(Idx()._d1(6)), 6 * D2);
    EXPECT_EQ(static_cast<int>(Idx()._d0(5)), 5 * D1 * D2);
    EXPECT_EQ(static_cast<int>(Idx()._d0(3)._d1(4)._d2(5)), 3 * (D1 * D2) + 4 * D2 + 5);
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

UTEST(Index4D, offset_from_index)
{
    constexpr int D1 = 77;
    constexpr int D2 = 88;
    constexpr int D3 = 99;
    using Idx = spio::Index4D<D1, D2, D3>;
    EXPECT_EQ(static_cast<int>(Idx()._d3(4)), 4);
    EXPECT_EQ(static_cast<int>(Idx()._d0(7)._d1(8)._d2(9)._d3(10)), 7 * (D1 * D2 * D3) + 8 * (D2 * D3) + 9 * D3 + 10);
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
