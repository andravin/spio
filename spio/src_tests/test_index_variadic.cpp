#include "utest.h"
#include "spio/index_variadic.h"

using namespace spio;

class I_Dim : public Dim<I_Dim>
{
public:
    using Dim<I_Dim>::Dim;
};

class J_Dim : public Dim<J_Dim>
{
public:
    using Dim<J_Dim>::Dim;
};

class K_Dim : public Dim<K_Dim>
{
public:
    using Dim<K_Dim>::Dim;
};

template <>
struct utest_type_deducer<I_Dim>
{
    static void _(const I_Dim d)
    {
        UTEST_PRINTF("%d", d.get());
    }
};

template <>
struct utest_type_deducer<J_Dim>
{
    static void _(const J_Dim d)
    {
        UTEST_PRINTF("%d", d.get());
    }
};

template <>
struct utest_type_deducer<K_Dim>
{
    static void _(const K_Dim d)
    {
        UTEST_PRINTF("%d", d.get());
    }
};

UTEST(Index1D, get)
{
    using Idx = Index<DimInfo<I_Dim, 16, 1>>;
    for (unsigned offset = 0; offset < 16; ++offset)
    {
        Idx idx(offset);
        EXPECT_EQ(idx.get<I_Dim>(), offset);
    }
}

UTEST(Index2D, get)
{
    // I x J matrix with size 32 x 8.
    using Idx = Index<DimInfo<I_Dim, 32, 8>, DimInfo<J_Dim, 8, 1>>;
    for (unsigned offset = 0; offset < 256; ++offset)
    {
        Idx idx(offset);
        EXPECT_EQ(idx.get<I_Dim>(), offset / 8);
        EXPECT_EQ(idx.get<J_Dim>(), offset % 8);
    }
}

UTEST(Index3D, get)
{
    // I x J x K tensor with size 16 x 8 x 4
    // Stride for K is 1
    // Stride for J is 4 (= size of K)
    // Stride for I is 32 (= size of J * stride of J)
    using Idx = Index<
        DimInfo<I_Dim, 16, 32>,
        DimInfo<J_Dim, 8, 4>,
        DimInfo<K_Dim, 4, 1>>;

    for (unsigned offset = 0; offset < 16 * 8 * 4; ++offset)
    {
        Idx idx(offset);
        EXPECT_EQ(idx.get<I_Dim>(), offset / (8 * 4));
        EXPECT_EQ(idx.get<J_Dim>(), (offset / 4) % 8);
        EXPECT_EQ(idx.get<K_Dim>(), offset % 4);
    }
}

UTEST(IndexSize, total_size)
{
    // 1D index
    using Idx1D = Index<DimInfo<I_Dim, 16, 1>>;
    EXPECT_EQ(Idx1D::total_size, 16);
    EXPECT_EQ(Idx1D::size(), 16);

    // 2D index
    using Idx2D = Index<DimInfo<I_Dim, 32, 8>, DimInfo<J_Dim, 8, 1>>;
    EXPECT_EQ(Idx2D::total_size, 32 * 8);
    EXPECT_EQ(Idx2D::size(), 32 * 8);

    // 3D index
    using Idx3D = Index<
        DimInfo<I_Dim, 16, 32>,
        DimInfo<J_Dim, 8, 4>,
        DimInfo<K_Dim, 4, 1>>;
    EXPECT_EQ(Idx3D::total_size, 16 * 8 * 4);
    EXPECT_EQ(Idx3D::size(), 16 * 8 * 4);
}
