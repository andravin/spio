#include "utest.h"
#include "spio/index.h"

UTEST(index, constructor)
{
    constexpr int D1 = 16;
    using Idx = spio::Index2D<D1>;
    EXPECT_EQ(static_cast<int>(Idx()._d1(10)), 10);
    EXPECT_EQ(static_cast<int>(Idx()._d0(9)._d1(10)), 9 * D1 + 10);
}

UTEST_MAIN()
