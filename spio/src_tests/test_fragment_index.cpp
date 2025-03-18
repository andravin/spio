#include "utest.h"
// #include "spio/fragment_load_index.h"
#include "spio/fragment_index.h"

using namespace spio;

class I : public Dim<I>
{
    using Dim::Dim;
};
class J : public Dim<J>
{
    using Dim::Dim;
};
class K : public Dim<K>
{
    using Dim::Dim;
};

using K8 = Fold<K, 8>;
using K2 = Fold<K, 2>;
using K2M4 = Module<K, 4, 2>;

using J2 = Fold<J, 2>;
using J8 = Fold<J, 8>;
using J2M4 = Module<J, 4, 2>;

// UTEST(MMA_A_M16_K8_F16_LoadIndex, indices)
// {
//     for (int lane = 0; lane < 32; ++lane)
//     {
//         EXPECT_EQ(spio::MMA_A_M16_K8_F16_LoadIndex(lane).i(), lane % 16);
//         EXPECT_EQ(spio::MMA_A_M16_K8_F16_LoadIndex(lane).k8(), 0);
//     }
// }

// UTEST(MMA_A_M16_K16_F16_LoadIndex, indices)
// {
//     for (int lane = 0; lane < 32; ++lane)
//     {
//         EXPECT_EQ(spio::MMA_A_M16_K16_F16_LoadIndex(lane).i(), lane % 16);
//         EXPECT_EQ(spio::MMA_A_M16_K16_F16_LoadIndex(lane).k8(), lane / 16);
//     }
// }

// UTEST(MMA_B_N8_K8_F16_LoadIndex, indices)
// {
//     for (int lane = 0; lane < 32; ++lane)
//     {
//         EXPECT_EQ(spio::MMA_B_N8_K8_F16_LoadIndex(lane).j(), lane % 8);
//         EXPECT_EQ(spio::MMA_B_N8_K8_F16_LoadIndex(lane).k8(), 0);
//     }
// }

// UTEST(MMA_B_N8_K16_F16_LoadIndex, indices)
// {
//     for (int lane = 0; lane < 32; ++lane)
//     {
//         EXPECT_EQ(spio::MMA_B_N8_K16_F16_LoadIndex(lane).j(), lane % 8);
//         EXPECT_EQ(spio::MMA_B_N8_K16_F16_LoadIndex(lane).k8(), (lane / 8) % 2);
//     }
// }

// UTEST(MMA_B_N16_K16_F16_LoadIndex, indices)
// {
//     for (int lane = 0; lane < 32; ++lane)
//     {
//         EXPECT_EQ(spio::MMA_B_N16_K16_F16_LoadIndex(lane).j(), (lane % 8) + lane / 16 * 8);
//         EXPECT_EQ(spio::MMA_B_N16_K16_F16_LoadIndex(lane).k8(), (lane / 8) % 2);
//     }
// }

UTEST(MMA_A_88_F16_Index, indices)
{
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int f = 0; f < 4; ++f)
        {
            MMA_A_88_F16_Index<I, K> idx(lane);
            EXPECT_EQ(idx.get<I>(f).get(), lane / 4 + (f % 2) * 8);
            EXPECT_EQ(idx.get<K2>(f).get(), (lane % 4) + f / 2 * 4);
            EXPECT_EQ(idx.get<K8>(f).get(), f / 2);
            EXPECT_EQ(idx.get<K2M4>().get(), lane % 4);
        }
    }
}

UTEST(MMA_B_88_F16_Index, indices)
{
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int f = 0; f < 4; ++f)
        {
            MMA_B_88_F16_Index<K, J> idx(lane);
            EXPECT_EQ(idx.get<J>(f).get(), (lane / 4) + (f / 2) * 8);
            EXPECT_EQ(idx.get<K2>(f).get(), (lane % 4) + (f % 2) * 4);
            EXPECT_EQ(idx.get<K8>(f).get(), f % 2);
            EXPECT_EQ(idx.get<K2M4>().get(), lane % 4);
        }
    }
}

UTEST(MMA_C_M16_N16_F32_Index, indices)
{
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int f = 0; f < 4; ++f)
        {
            MMA_C_88_F32_Index<I, J> idx(lane);
            EXPECT_EQ(idx.get<I>(f).get(), (lane / 4) + (f % 2) * 8);
            EXPECT_EQ(idx.get<J2>(f).get(), (lane % 4) + f / 2 * 4);
            EXPECT_EQ(idx.get<J8>(f).get(), f / 2);
            EXPECT_EQ(idx.get<J2M4>().get(), lane % 4);
        }
    }
}