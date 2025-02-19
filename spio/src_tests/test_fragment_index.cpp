#include "utest.h"
#include "spio/fragment_load_index.h"
#include "spio/fragment_index.h"

UTEST(MMA_A_M16_K8_F16_LoadIndex, indices)
{
    for (int lane = 0; lane < 32; ++lane)
    {
        EXPECT_EQ(spio::MMA_A_M16_K8_F16_LoadIndex(lane).i(), lane % 16);
        EXPECT_EQ(spio::MMA_A_M16_K8_F16_LoadIndex(lane).k8(), 0);
    }
}

UTEST(MMA_A_M16_K16_F16_LoadIndex, indices)
{
    for (int lane = 0; lane < 32; ++lane)
    {
        EXPECT_EQ(spio::MMA_A_M16_K16_F16_LoadIndex(lane).i(), lane % 16);
        EXPECT_EQ(spio::MMA_A_M16_K16_F16_LoadIndex(lane).k8(), lane / 16);
    }
}

UTEST(MMA_B_N8_K8_F16_LoadIndex, indices)
{
    for (int lane = 0; lane < 32; ++lane)
    {
        EXPECT_EQ(spio::MMA_B_N8_K8_F16_LoadIndex(lane).j(), lane % 8);
        EXPECT_EQ(spio::MMA_B_N8_K8_F16_LoadIndex(lane).k8(), 0);
    }
}

UTEST(MMA_B_N8_K16_F16_LoadIndex, indices)
{
    for (int lane = 0; lane < 32; ++lane)
    {
        EXPECT_EQ(spio::MMA_B_N8_K16_F16_LoadIndex(lane).j(), lane % 8);
        EXPECT_EQ(spio::MMA_B_N8_K16_F16_LoadIndex(lane).k8(), (lane / 8) % 2);
    }
}

UTEST(MMA_B_N16_K16_F16_LoadIndex, indices)
{
    for (int lane = 0; lane < 32; ++lane)
    {
        EXPECT_EQ(spio::MMA_B_N16_K16_F16_LoadIndex(lane).j(), (lane % 8) + lane / 16 * 8);
        EXPECT_EQ(spio::MMA_B_N16_K16_F16_LoadIndex(lane).k8(), (lane / 8) % 2);
    }
}

UTEST(MMA_A_88_F16_Index, indices)
{
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int idx = 0; idx < 4; ++idx)
        {
            EXPECT_EQ(spio::MMA_A_88_F16_Index(lane).i(idx), lane / 4 + (idx % 2) * 8);
            EXPECT_EQ(spio::MMA_A_88_F16_Index(lane).k2(idx), (lane % 4) + idx / 2 * 4);
            EXPECT_EQ(spio::MMA_A_88_F16_Index(lane).k8(idx), idx / 2);
            EXPECT_EQ(spio::MMA_A_88_F16_Index(lane).k2m4(), lane % 4);
        }
    }
}

UTEST(MMA_B_88_F16_Index, indices)
{
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int idx = 0; idx < 4; ++idx)
        {
            EXPECT_EQ(spio::MMA_B_88_F16_Index(lane).j(idx), (lane / 4) + (idx / 2) * 8);
            EXPECT_EQ(spio::MMA_B_88_F16_Index(lane).k2(idx), (lane % 4) + (idx % 2) * 4);
            EXPECT_EQ(spio::MMA_B_88_F16_Index(lane).k8(idx), idx % 2);
            EXPECT_EQ(spio::MMA_B_88_F16_Index(lane).k2m4(), lane % 4);
        }
    }
}

UTEST(MMA_C_M16_N16_F32_Index, indices)
{
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int idx = 0; idx < 4; ++idx)
        {
            EXPECT_EQ(spio::MMA_C_88_F32_Index(lane).i(idx), (lane / 4) + (idx % 2) * 8);
            EXPECT_EQ(spio::MMA_C_88_F32_Index(lane).j2(idx), (lane % 4) + idx / 2 * 4);
            EXPECT_EQ(spio::MMA_C_88_F32_Index(lane).j8(idx), idx / 2);
            EXPECT_EQ(spio::MMA_C_88_F32_Index(lane).j2m4(), lane % 4);
        }
    }
}