#include "utest.h"
#include "spio/tensor.h"
#include "spio/dim.h"

namespace
{
    class D0 : public spio::Dim
    {
    public:
        using spio::Dim::Dim;
        D0 operator-(D0 i) const { return D0(get() - i.get()); }
        D0 operator+(D0 i) const { return D0(get() + i.get()); }
        bool operator==(D0 i) const { return get() == i.get(); }
    };

    class D1 : public spio::Dim
    {
    public:
        using spio::Dim::Dim;
        D1 operator-(D1 i) const { return D1(get() - i.get()); }
        D1 operator+(D1 i) const { return D1(get() + i.get()); }
        bool operator==(D1 i) const { return get() == i.get(); }
    };

    class D2 : public spio::Dim
    {
    public:
        using spio::Dim::Dim;
        D2 operator-(D2 i) const { return D2(get() - i.get()); }
        D2 operator+(D2 i) const { return D2(get() + i.get()); }
        bool operator==(D2 i) const { return get() == i.get(); }
    };

    class D3 : public spio::Dim
    {
    public:
        using spio::Dim::Dim;
        D3 operator-(D3 i) const { return D3(get() - i.get()); }
        D3 operator+(D3 i) const { return D3(get() + i.get()); }
        bool operator==(D3 i) const { return get() == i.get(); }
    };

    class D4 : public spio::Dim
    {
    public:
        using spio::Dim::Dim;
        D4 operator-(D4 i) const { return D4(get() - i.get()); }
        D4 operator+(D4 i) const { return D4(get() + i.get()); }
        bool operator==(D4 i) const { return get() == i.get(); }
    };

    class D5 : public spio::Dim
    {
    public:
        using spio::Dim::Dim;
        D5 operator-(D5 i) const { return D5(get() - i.get()); }
        D5 operator+(D5 i) const { return D5(get() + i.get()); }
        bool operator==(D5 i) const { return get() == i.get(); }
    };

    class D6 : public spio::Dim
    {
    public:
        using spio::Dim::Dim;
        D6 operator-(D6 i) const { return D6(get() - i.get()); }
        D6 operator+(D6 i) const { return D6(get() + i.get()); }
        bool operator==(D6 i) const { return get() == i.get(); }
    };
}

struct Tensor1D_Fixture
{
    using Tensor = spio::Tensor1D<float, D0, 8>;
    float data[Tensor::size];
    Tensor tensor;
};

UTEST_F_SETUP(Tensor1D_Fixture)
{
    utest_fixture->tensor = Tensor1D_Fixture::Tensor(utest_fixture->data);
}

UTEST_F_TEARDOWN(Tensor1D_Fixture)
{
}

struct Tensor2D_Fixture
{
    using Tensor = spio::Tensor2D<float, D0, D1, 8, 16>;
    float data[Tensor::size];
    Tensor tensor;
};

UTEST_F_SETUP(Tensor2D_Fixture)
{
    utest_fixture->tensor = Tensor2D_Fixture::Tensor(utest_fixture->data);
}

UTEST_F_TEARDOWN(Tensor2D_Fixture)
{
}

struct Tensor3D_Fixture
{
    using Tensor = spio::Tensor3D<float, D0, D1, D2, 8, 16, 4>;
    float data[Tensor::size];
    Tensor tensor;
};

UTEST_F_SETUP(Tensor3D_Fixture)
{
    utest_fixture->tensor = Tensor3D_Fixture::Tensor(utest_fixture->data);
}

UTEST_F_TEARDOWN(Tensor3D_Fixture)
{
}

struct Tensor4D_Fixture
{
    using Tensor = spio::Tensor4D<float, D0, D1, D2, D3, 8, 16, 4, 2>;
    float data[Tensor::size];
    Tensor tensor;
};

UTEST_F_SETUP(Tensor4D_Fixture)
{
    utest_fixture->tensor = Tensor4D_Fixture::Tensor(utest_fixture->data);
}

UTEST_F_TEARDOWN(Tensor4D_Fixture)
{
}

struct Tensor5D_Fixture
{
    using Tensor = spio::Tensor5D<float, D0, D1, D2, D3, D4, 8, 16, 32, 4, 2>;
    float data[Tensor::size];
    Tensor tensor;
};

UTEST_F_SETUP(Tensor5D_Fixture)
{
    utest_fixture->tensor = Tensor5D_Fixture::Tensor(utest_fixture->data);
}

UTEST_F_TEARDOWN(Tensor5D_Fixture)
{
}

struct Tensor6D_Fixture
{
    using Tensor = spio::Tensor6D<float, D0, D1, D2, D3, D4, D5, 8, 16, 32, 4, 2, 8>;
    float data[Tensor::size];
    Tensor tensor;
};

UTEST_F_SETUP(Tensor6D_Fixture)
{
    utest_fixture->tensor = Tensor6D_Fixture::Tensor(utest_fixture->data);
}

UTEST_F_TEARDOWN(Tensor6D_Fixture)
{
}

struct Tensor7D_Fixture
{
    using Tensor = spio::Tensor7D<float, D0, D1, D2, D3, D4, D5, D6, 8, 16, 32, 4, 2, 8, 3>;
    float data[Tensor::size];
    Tensor tensor;
};

UTEST_F_SETUP(Tensor7D_Fixture)
{
    utest_fixture->tensor = Tensor7D_Fixture::Tensor(utest_fixture->data);
}

UTEST_F_TEARDOWN(Tensor7D_Fixture)
{
}

UTEST_F(Tensor1D_Fixture, indexing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0))
    {
        auto p1 = tensor[d0].get();
        auto p2 = &data[d0.get()];
        EXPECT_EQ(p1, p2);
    }
}

UTEST_F(Tensor1D_Fixture, slicing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0 - D0(2)))
    {
        auto slice = tensor.slice<2>(d0);
        EXPECT_TRUE(slice.size_0 == D0(2));
        for (auto s0 : spio::range(slice.size_0))
        {
            EXPECT_EQ(slice[s0].get(), &data[d0.get() + s0.get()]);
        }
        for (int i = 0; i < 2; ++i)
        {
            EXPECT_EQ(slice[D0(i)].get(), &data[d0.get() + i]);
        }
    }
}

UTEST_F(Tensor2D_Fixture, indexing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0))
    {
        for (auto d1 : spio::range(tensor.size_1))
        {
            auto p1 = tensor[d0][d1].get();
            auto p2 = &data[d0.get() * tensor.stride_0 + d1.get() * tensor.stride_1];
            EXPECT_EQ(p1, p2);
        }
    }
}

UTEST_F(Tensor2D_Fixture, slicing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0 - D0(2)))
    {
        auto slice = tensor.slice<2>(d0);
        auto p2 = &data[d0.get() * tensor.stride_0];
        EXPECT_EQ(slice.get(), p2);
        EXPECT_TRUE(slice.size_0 == D0(2));
    }
    for (auto d1 : spio::range(tensor.size_1 - D1(2)))
    {
        auto slice = tensor.slice<2>(d1);
        auto p2 = &data[d1.get() * tensor.stride_1];
        EXPECT_EQ(slice.get(), p2);
        EXPECT_TRUE(slice.size_1 == D1(2));
    }
}

UTEST_F(Tensor2D_Fixture, nested_slicing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0 - D0(2)))
    {
        for (auto d1 : spio::range(tensor.size_1 - D1(4)))
        {
            auto slice = tensor.slice<2>(d0).slice<4>(d1);
            EXPECT_TRUE(slice.size_0 == D0(2));
            EXPECT_TRUE(slice.size_1 == D1(4));
            for (auto s0 : spio::range(slice.size_0))
            {
                for (auto s1 : spio::range(slice.size_1))
                {
                    EXPECT_EQ(slice[s0][s1].get(), &data[(d0 + s0).get() * tensor.stride_0 + (d1 + s1).get() * tensor.stride_1]);
                }
            }
        }
    }
}

UTEST_F(Tensor3D_Fixture, indexing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0))
    {
        for (auto d1 : spio::range(tensor.size_1))
        {
            for (auto d2 : spio::range(tensor.size_2))
            {
                auto p1 = tensor[d0][d1][d2].get();
                auto p2 = &data[d0.get() * tensor.stride_0 +
                                d1.get() * tensor.stride_1 +
                                d2.get() * tensor.stride_2];
                EXPECT_EQ(p1, p2);
            }
        }
    }
}

UTEST_F(Tensor3D_Fixture, nested_slicing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0 - D0(2)))
    {
        for (auto d1 : spio::range(tensor.size_1 - D1(2)))
        {
            for (auto d2 : spio::range(tensor.size_2 - D2(4)))
            {
                auto slice = tensor.slice<2>(d0).slice<2>(d1).slice<4>(d2);
                EXPECT_TRUE(slice.size_0 == D0(2));
                EXPECT_TRUE(slice.size_1 == D1(2));
                EXPECT_TRUE(slice.size_2 == D2(4));
                for (auto s0 : spio::range(slice.size_0))
                {
                    for (auto s1 : spio::range(slice.size_1))
                    {
                        for (auto s2 : spio::range(slice.size_2))
                        {
                            EXPECT_EQ(
                                slice[s0][s1][s2].get(),
                                &data[(d0 + s0).get() * tensor.stride_0 +
                                      (d1 + s1).get() * tensor.stride_1 +
                                      (d2 + s2).get() * tensor.stride_2]);
                        }
                    }
                }
            }
        }
    }
}

UTEST_F(Tensor4D_Fixture, indexing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0))
    {
        for (auto d1 : spio::range(tensor.size_1))
        {
            for (auto d2 : spio::range(tensor.size_2))
            {
                for (auto d3 : spio::range(tensor.size_3))
                {
                    auto p1 = tensor[d0][d1][d2][d3].get();
                    auto p2 = &data[d0.get() * tensor.stride_0 +
                                    d1.get() * tensor.stride_1 +
                                    d2.get() * tensor.stride_2 +
                                    d3.get() * tensor.stride_3];
                    EXPECT_EQ(p1, p2);
                }
            }
        }
    }
}

UTEST_F(Tensor4D_Fixture, nested_slicing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0 - D0(2)))
    {
        for (auto d1 : spio::range(tensor.size_1 - D1(2)))
        {
            for (auto d2 : spio::range(tensor.size_2 - D2(1)))
            {
                for (auto d3 : spio::range(tensor.size_3 - D3(1)))
                {
                    auto slice = tensor.slice<2>(d0).slice<2>(d1).slice<1>(d2).slice<1>(d3);
                    EXPECT_TRUE(slice.size_0 == D0(2));
                    EXPECT_TRUE(slice.size_1 == D1(2));
                    EXPECT_TRUE(slice.size_2 == D2(1));
                    EXPECT_TRUE(slice.size_3 == D3(1));
                    for (auto s0 : spio::range(slice.size_0))
                    {
                        for (auto s1 : spio::range(slice.size_1))
                        {
                            for (auto s2 : spio::range(slice.size_2))
                            {
                                for (auto s3 : spio::range(slice.size_3))
                                {
                                    EXPECT_EQ(
                                        slice[s0][s1][s2][s3].get(),
                                        &data[(d0 + s0).get() * tensor.stride_0 +
                                              (d1 + s1).get() * tensor.stride_1 +
                                              (d2 + s2).get() * tensor.stride_2 +
                                              (d3 + s3).get() * tensor.stride_3]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

UTEST_F(Tensor5D_Fixture, indexing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0))
    {
        for (auto d1 : spio::range(tensor.size_1))
        {
            for (auto d2 : spio::range(tensor.size_2))
            {
                for (auto d3 : spio::range(tensor.size_3))
                {
                    for (auto d4 : spio::range(tensor.size_4))
                    {
                        auto p1 = tensor[d0][d1][d2][d3][d4].get();
                        auto p2 = &data[d0.get() * tensor.stride_0 +
                                        d1.get() * tensor.stride_1 +
                                        d2.get() * tensor.stride_2 +
                                        d3.get() * tensor.stride_3 +
                                        d4.get() * tensor.stride_4];
                        EXPECT_EQ(p1, p2);
                    }
                }
            }
        }
    }
}

UTEST_F(Tensor5D_Fixture, nested_slicing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0 - D0(2)))
    {
        for (auto d1 : spio::range(tensor.size_1 - D1(2)))
        {
            for (auto d2 : spio::range(tensor.size_2 - D2(1)))
            {
                for (auto d3 : spio::range(tensor.size_3 - D3(1)))
                {
                    for (auto d4 : spio::range(tensor.size_4 - D4(1)))
                    {
                        auto slice = tensor.slice<2>(d0).slice<2>(d1).slice<1>(d2).slice<1>(d3).slice<1>(d4);
                        EXPECT_TRUE(slice.size_0 == D0(2));
                        EXPECT_TRUE(slice.size_1 == D1(2));
                        EXPECT_TRUE(slice.size_2 == D2(1));
                        EXPECT_TRUE(slice.size_3 == D3(1));
                        EXPECT_TRUE(slice.size_4 == D4(1));
                        for (auto s0 : spio::range(slice.size_0))
                        {
                            for (auto s1 : spio::range(slice.size_1))
                            {
                                for (auto s2 : spio::range(slice.size_2))
                                {
                                    for (auto s3 : spio::range(slice.size_3))
                                    {
                                        for (auto s4 : spio::range(slice.size_4))
                                        {
                                            EXPECT_EQ(
                                                slice[s0][s1][s2][s3][s4].get(),
                                                &data[(d0 + s0).get() * tensor.stride_0 +
                                                      (d1 + s1).get() * tensor.stride_1 +
                                                      (d2 + s2).get() * tensor.stride_2 +
                                                      (d3 + s3).get() * tensor.stride_3 +
                                                      (d4 + s4).get() * tensor.stride_4]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

UTEST_F(Tensor6D_Fixture, indexing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0))
    {
        for (auto d1 : spio::range(tensor.size_1))
        {
            for (auto d2 : spio::range(tensor.size_2))
            {
                for (auto d3 : spio::range(tensor.size_3))
                {
                    for (auto d4 : spio::range(tensor.size_4))
                    {
                        for (auto d5 : spio::range(tensor.size_5))
                        {
                            auto p1 = tensor[d0][d1][d2][d3][d4][d5].get();
                            auto p2 = &data[d0.get() * tensor.stride_0 +
                                            d1.get() * tensor.stride_1 +
                                            d2.get() * tensor.stride_2 +
                                            d3.get() * tensor.stride_3 +
                                            d4.get() * tensor.stride_4 +
                                            d5.get() * tensor.stride_5];
                            EXPECT_EQ(p1, p2);
                        }
                    }
                }
            }
        }
    }
}

UTEST_F(Tensor6D_Fixture, nested_slicing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0 - D0(2)))
    {
        for (auto d1 : spio::range(tensor.size_1 - D1(2)))
        {
            for (auto d2 : spio::range(tensor.size_2 - D2(1)))
            {
                for (auto d3 : spio::range(tensor.size_3 - D3(1)))
                {
                    for (auto d4 : spio::range(tensor.size_4 - D4(1)))
                    {
                        for (auto d5 : spio::range(tensor.size_5 - D5(1)))
                        {
                            auto slice = tensor.slice<2>(d0).slice<2>(d1).slice<1>(d2).slice<1>(d3).slice<1>(d4).slice<1>(d5);
                            EXPECT_TRUE(slice.size_0 == D0(2));
                            EXPECT_TRUE(slice.size_1 == D1(2));
                            EXPECT_TRUE(slice.size_2 == D2(1));
                            EXPECT_TRUE(slice.size_3 == D3(1));
                            EXPECT_TRUE(slice.size_4 == D4(1));
                            EXPECT_TRUE(slice.size_5 == D5(1));
                            for (auto s0 : spio::range(slice.size_0))
                            {
                                for (auto s1 : spio::range(slice.size_1))
                                {
                                    for (auto s2 : spio::range(slice.size_2))
                                    {
                                        for (auto s3 : spio::range(slice.size_3))
                                        {
                                            for (auto s4 : spio::range(slice.size_4))
                                            {
                                                for (auto s5 : spio::range(slice.size_5))
                                                {
                                                    EXPECT_EQ(
                                                        slice[s0][s1][s2][s3][s4][s5].get(),
                                                        &data[(d0 + s0).get() * tensor.stride_0 +
                                                              (d1 + s1).get() * tensor.stride_1 +
                                                              (d2 + s2).get() * tensor.stride_2 +
                                                              (d3 + s3).get() * tensor.stride_3 +
                                                              (d4 + s4).get() * tensor.stride_4 +
                                                              (d5 + s5).get() * tensor.stride_5]);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

UTEST_F(Tensor7D_Fixture, indexing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0))
    {
        for (auto d1 : spio::range(tensor.size_1))
        {
            for (auto d2 : spio::range(tensor.size_2))
            {
                for (auto d3 : spio::range(tensor.size_3))
                {
                    for (auto d4 : spio::range(tensor.size_4))
                    {
                        for (auto d5 : spio::range(tensor.size_5))
                        {
                            for (auto d6 : spio::range(tensor.size_6))
                            {
                                auto p1 = tensor[d0][d1][d2][d3][d4][d5][d6].get();
                                auto p2 = &data[d0.get() * tensor.stride_0 +
                                                d1.get() * tensor.stride_1 +
                                                d2.get() * tensor.stride_2 +
                                                d3.get() * tensor.stride_3 +
                                                d4.get() * tensor.stride_4 +
                                                d5.get() * tensor.stride_5 +
                                                d6.get() * tensor.stride_6];
                                EXPECT_EQ(p1, p2);
                            }
                        }
                    }
                }
            }
        }
    }
}

UTEST_F(Tensor7D_Fixture, nested_slicing)
{
    auto &tensor = utest_fixture->tensor;
    auto &data = utest_fixture->data;
    for (auto d0 : spio::range(tensor.size_0 - D0(2)))
    {
        for (auto d1 : spio::range(tensor.size_1 - D1(2)))
        {
            for (auto d2 : spio::range(tensor.size_2 - D2(1)))
            {
                for (auto d3 : spio::range(tensor.size_3 - D3(1)))
                {
                    for (auto d4 : spio::range(tensor.size_4 - D4(1)))
                    {
                        for (auto d5 : spio::range(tensor.size_5 - D5(1)))
                        {
                            for (auto d6 : spio::range(tensor.size_6 - D6(1)))
                            {
                                auto slice = tensor.slice<2>(d0).slice<2>(d1).slice<1>(d2).slice<1>(d3).slice<1>(d4).slice<1>(d5).slice<1>(d6);
                                EXPECT_TRUE(slice.size_0 == D0(2));
                                EXPECT_TRUE(slice.size_1 == D1(2));
                                EXPECT_TRUE(slice.size_2 == D2(1));
                                EXPECT_TRUE(slice.size_3 == D3(1));
                                EXPECT_TRUE(slice.size_4 == D4(1));
                                EXPECT_TRUE(slice.size_5 == D5(1));
                                EXPECT_TRUE(slice.size_6 == D6(1));
                                for (auto s0 : spio::range(slice.size_0))
                                {
                                    for (auto s1 : spio::range(slice.size_1))
                                    {
                                        for (auto s2 : spio::range(slice.size_2))
                                        {
                                            for (auto s3 : spio::range(slice.size_3))
                                            {
                                                for (auto s4 : spio::range(slice.size_4))
                                                {
                                                    for (auto s5 : spio::range(slice.size_5))
                                                    {
                                                        for (auto s6 : spio::range(slice.size_6))
                                                        {
                                                            EXPECT_EQ(
                                                                slice[s0][s1][s2][s3][s4][s5][s6].get(),
                                                                &data[(d0 + s0).get() * tensor.stride_0 +
                                                                      (d1 + s1).get() * tensor.stride_1 +
                                                                      (d2 + s2).get() * tensor.stride_2 +
                                                                      (d3 + s3).get() * tensor.stride_3 +
                                                                      (d4 + s4).get() * tensor.stride_4 +
                                                                      (d5 + s5).get() * tensor.stride_5 +
                                                                      (d6 + s6).get() * tensor.stride_6]);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}