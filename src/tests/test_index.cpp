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

UTEST_MAIN()
