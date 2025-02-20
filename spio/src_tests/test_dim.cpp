#include "utest.h"

#include "spio/dim.h"

UTEST(Dim, accessors)
{
    EXPECT_EQ(spio::Dim(7).get(), 7);
}

namespace
{
    class IDim : public spio::Dim
    {
    public:
        using Base = spio::Dim;
        using Base::Base;
        template <class NewDimType>
        DEVICE constexpr auto cast() const -> NewDimType
        {
            {
                return NewDimType(Base::get());
            }
        }
        template <unsigned Stride>
        DEVICE constexpr spio::Fold<IDim, Stride> fold() const
        {
            {
                return spio::Fold<IDim, Stride>(*this);
            }
        }
        bool operator<(const IDim &other) const { return Base::operator<(other); }
        bool operator>(const IDim &other) const { return other < *this; }
        bool operator<=(const IDim &other) const { return !(*this > other); }
        bool operator>=(const IDim &other) const { return !(*this < other); }
        bool operator==(const IDim &other) const { return Base::operator==(other); }
        bool operator!=(const IDim &other) const { return !(*this == other); }
        IDim operator+(const IDim &other) const { return IDim(Base::_add(other)); }
        IDim operator-(const IDim &other) const { return IDim(Base::_sub(other)); }
        IDim operator%(const IDim &other) const { return IDim(Base::_modulus(other)); }
    };

    class JDim : public spio::Dim
    {
    public:
        using Base = spio::Dim;
        using Base::Base;
        bool operator==(const JDim &other) const { return Base::operator==(other); }
    };
}

UTEST(Dim, subclass_methods)
{
    EXPECT_EQ(IDim(7).get(), 7);
    EXPECT_TRUE(IDim(7) == IDim(7));
    EXPECT_TRUE(IDim(7) < IDim(8));
    EXPECT_TRUE(IDim(7) <= IDim(8));
    EXPECT_TRUE(IDim(8) > IDim(7));
    EXPECT_TRUE(IDim(8) >= IDim(7));
    EXPECT_TRUE(IDim(7) == IDim(7));
    EXPECT_TRUE(IDim(7) <= IDim(7));
    EXPECT_TRUE(IDim(7) >= IDim(7));
    EXPECT_TRUE(IDim(7) != IDim(8));
    EXPECT_TRUE(IDim(7) + IDim(8) == IDim(15));
    EXPECT_TRUE(IDim(8) - IDim(7) == IDim(1));
    EXPECT_TRUE(IDim(7) - IDim(8) == IDim(-1));
    EXPECT_TRUE(IDim(8) + IDim(-1) == IDim(7));
    EXPECT_TRUE(IDim(8) % IDim(3) == IDim(2));
}

UTEST(IDim, fold)
{
    using F8 = spio::Fold<IDim, 8>;
    EXPECT_TRUE(IDim(32).fold<8>() == F8(4));
}

UTEST(IDim, cast)
{
    EXPECT_TRUE(IDim(8).cast<JDim>() == JDim(8));
}

UTEST(Fold, methods)
{
    using F8 = spio::Fold<spio::Dim, 8>;
    using F16 = spio::Fold<spio::Dim, 16>;
    EXPECT_EQ(F8(32).get(), 32);
    EXPECT_EQ(F8(32).unfold().get(), 8 * 32);
    EXPECT_EQ(F8(32).fold<16>().get(), 32 * 8 / 16);
    EXPECT_TRUE(F8(32).fold<16>() == F16(16));
    EXPECT_TRUE(F16(16).fold<8>() == F8(32));
    EXPECT_TRUE(F8(32) < F8(33));
    EXPECT_TRUE(F8(32) <= F8(33));
    EXPECT_TRUE(F8(33) > F8(32));
    EXPECT_TRUE(F8(33) >= F8(32));
    EXPECT_TRUE(F8(32) == F8(32));
    EXPECT_TRUE(F8(32) <= F8(32));
    EXPECT_TRUE(F8(32) >= F8(32));
    EXPECT_TRUE(F8(32) + F8(16) == F8(48));
    EXPECT_TRUE(F8(32) - F8(16) == F8(16));
}

UTEST(Fold, cast)
{
    using I32 = spio::Fold<IDim, 32>;
    using J32 = spio::Fold<JDim, 32>;
    EXPECT_TRUE(I32(8).cast<JDim>() == J32(8));
}

UTEST(Fold, dim_copy_constructor)
{
    using I32 = spio::Fold<IDim, 32>;
    EXPECT_TRUE(I32(IDim(256)) == I32(8));
}

UTEST(Fold, stride)
{
    using I32 = spio::Fold<IDim, 32>;
    EXPECT_TRUE(I32::stride == IDim(32));
}

UTEST(Fold, data_type)
{
    using I32 = spio::Fold<IDim, 32>;
    EXPECT_TRUE(I32::dim_type(8) == IDim(8));
}
