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

UTEST_MAIN()
