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

UTEST(Index5D, index_from_offset)
{
    constexpr int D1 = 45;
    constexpr int D2 = 56;
    constexpr int D3 = 67;
    constexpr int D4 = 78;
    using Idx = spio::Index5D<D1, D2, D3, D4>;
    constexpr int Offset = 867539;
    Idx idx(Offset);
    EXPECT_EQ(idx._d4(), Offset % D4);
    EXPECT_EQ(idx._d3(), (Offset / D4) % D3);
    EXPECT_EQ(idx._d2(), (Offset / (D3 * D4)) % D2);
    EXPECT_EQ(idx._d1(), (Offset / (D2 * D3 * D4)) % D1);
    EXPECT_EQ(idx._d0(), Offset / (D1 * D2 * D3 * D4));
}

UTEST(Index6D, index_from_offset)
{
    constexpr int D1 = 45;
    constexpr int D2 = 56;
    constexpr int D3 = 67;
    constexpr int D4 = 78;
    constexpr int D5 = 89;
    using Idx = spio::Index6D<D1, D2, D3, D4, D5>;
    constexpr int Offset = 867539;
    Idx idx(Offset);
    EXPECT_EQ(idx._d5(), Offset % D5);
    EXPECT_EQ(idx._d4(), (Offset / D5) % D4);
    EXPECT_EQ(idx._d3(), (Offset / (D4 * D5)) % D3);
    EXPECT_EQ(idx._d2(), (Offset / (D3 * D4 * D5)) % D2);
    EXPECT_EQ(idx._d1(), (Offset / (D2 * D3 * D4 * D5)) % D1);
    EXPECT_EQ(idx._d0(), Offset / (D1 * D2 * D3 * D4 * D5));
}

UTEST(Index7D, index_from_offset)
{
    constexpr int D1 = 45;
    constexpr int D2 = 56;
    constexpr int D3 = 67;
    constexpr int D4 = 78;
    constexpr int D5 = 89;
    constexpr int D6 = 90;
    using Idx = spio::Index7D<D1, D2, D3, D4, D5, D6>;
    constexpr int Offset = 867539;
    Idx idx(Offset);
    EXPECT_EQ(idx._d6(), Offset % D6);
    EXPECT_EQ(idx._d5(), (Offset / D6) % D5);
    EXPECT_EQ(idx._d4(), (Offset / (D5 * D6)) % D4);
    EXPECT_EQ(idx._d3(), (Offset / (D4 * D5 * D6)) % D3);
    EXPECT_EQ(idx._d2(), (Offset / (D3 * D4 * D5 * D6)) % D2);
    EXPECT_EQ(idx._d1(), (Offset / (D2 * D3 * D4 * D5 * D6)) % D1);
    EXPECT_EQ(idx._d0(), Offset / (D1 * D2 * D3 * D4 * D5 * D6));
}

UTEST_MAIN()
