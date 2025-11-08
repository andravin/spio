#include "utest.h"
#include "spio/strip_loader_params.h"

UTEST(StripLoaderParams, two_loads)
{
    constexpr int X16_tile = 8;
    constexpr int C16_tile = 2;
    constexpr int num_warps = 8;

    using Params = spio::StripLoaderParams<X16_tile, C16_tile, num_warps>;

    EXPECT_EQ(Params::active_warps, 8);
    EXPECT_EQ(Params::num_loads, 2);
}

UTEST(StripLoaderParams, one_load)
{
    constexpr int X16_tile = 8;
    constexpr int C16_tile = 1;
    constexpr int num_warps = 8;

    using Params = spio::StripLoaderParams<X16_tile, C16_tile, num_warps>;

    EXPECT_EQ(Params::active_warps, 8);
    EXPECT_EQ(Params::num_loads, 1);
}


UTEST(StripLoaderParams, one_load_cross_minor_axis)
{
    constexpr int X16_tile = 4;
    constexpr int C16_tile = 2;
    constexpr int num_warps = 8;

    using Params = spio::StripLoaderParams<X16_tile, C16_tile, num_warps>;

    EXPECT_EQ(Params::active_warps, 8);
    EXPECT_EQ(Params::num_loads, 1);
}

UTEST(StripLoaderParams, two_loads_idle_warps)
{
    constexpr int X16_tile = 5;
    constexpr int C16_tile = 2;
    constexpr int num_warps = 8;

    using Params = spio::StripLoaderParams<X16_tile, C16_tile, num_warps>;

    EXPECT_EQ(Params::active_warps, 5);
    EXPECT_EQ(Params::num_loads, 2);
}

