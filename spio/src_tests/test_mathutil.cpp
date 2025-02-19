#include "utest.h"
#include "spio/mathutil.h"

UTEST(mathutil, min)
{
    EXPECT_EQ(spio::min(1, 2), 1);
    EXPECT_EQ(spio::min(2, 1), 1);
    EXPECT_EQ(spio::min(1, 1), 1);
}

UTEST(mathutil, max)
{
    EXPECT_EQ(spio::max(1, 2), 2);
    EXPECT_EQ(spio::max(2, 1), 2);
    EXPECT_EQ(spio::max(1, 1), 1);
}

UTEST(mathutil, divup)
{
    EXPECT_EQ(spio::divup(1, 2), 1);
    EXPECT_EQ(spio::divup(7, 2), 4);
    EXPECT_EQ(spio::divup(8, 2), 4);
    EXPECT_EQ(spio::divup(9, 2), 5);
}
