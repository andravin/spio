#include "dim_test_types.h"
#include "spio/dim.h"

#include <array>

UTEST(Dim, subclass_methods) {
    EXPECT_EQ(I(7).get(), 7);
    EXPECT_TRUE(I(7) == I(7));
    EXPECT_TRUE(I(7) < I(8));
    EXPECT_TRUE(I(7) <= I(8));
    EXPECT_TRUE(I(8) > I(7));
    EXPECT_TRUE(I(8) >= I(7));
    EXPECT_TRUE(I(7) == I(7));
    EXPECT_TRUE(I(7) <= I(7));
    EXPECT_TRUE(I(7) >= I(7));
    EXPECT_TRUE(I(7) != I(8));
    EXPECT_TRUE(I(7) + I(8) == I(15));
    EXPECT_TRUE(I(8) - I(7) == I(1));
    EXPECT_TRUE(I(7) - I(8) == I(-1));
    EXPECT_TRUE(I(8) + I(-1) == I(7));
    EXPECT_TRUE(I(8) % I(3) == I(2));
}

UTEST(I, fold) {
    using F8 = spio::Fold<I, 8>;
    EXPECT_TRUE(I(32).fold<8>() == F8(4));
}

UTEST(I, cast) {
    EXPECT_TRUE(I(8).cast<J>() == J(8));
}

UTEST(Fold, methods) {
    using F8 = spio::Fold<I, 8>;
    using F16 = spio::Fold<I, 16>;
    EXPECT_EQ(F8(32).get(), 32);
    EXPECT_EQ(F8(32).unfold().get(), 8 * 32);
    EXPECT_EQ(F8(32).fold<16>().get(), 32 * 8 / 16);
    EXPECT_EQ(F8(16), F16(F8(16)));
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

UTEST(Fold, mixed_stride_addition) {
    using I8 = spio::Fold<I, 8>;
    using I16 = spio::Fold<I, 16>;
    EXPECT_TRUE(I8(8) + I16(4) == I8(16));
    EXPECT_TRUE(I16(4) + I8(8) == I8(16));
}

UTEST(Fold, dim_addition) {
    using I8 = spio::Fold<I, 8>;
    EXPECT_TRUE(I8(8) + I(32) == I(96));
    EXPECT_TRUE(I(32) + I8(8) == I(96));
}

UTEST(Fold, mixed_stride_subtraction) {
    using I8 = spio::Fold<I, 8>;
    using I16 = spio::Fold<I, 16>;
    EXPECT_TRUE(I8(16) - I16(4) == I8(8));
    EXPECT_TRUE(I16(4) - I8(8) == I8(0));
}

UTEST(Fold, dim_subtraction) {
    using I8 = spio::Fold<I, 8>;
    EXPECT_TRUE(I8(8) - I(32) == I(32));
    EXPECT_TRUE(I(96) - I8(8) == I(32));
}

UTEST(Fold, mix_stride_comparison) {
    using I8 = spio::Fold<I, 8>;
    using I16 = spio::Fold<I, 16>;
    EXPECT_TRUE(I8(8) < I16(8));
    EXPECT_TRUE(I16(2) < I8(8));
    EXPECT_TRUE(I8(8) <= I16(8));
    EXPECT_TRUE(I8(8) <= I16(4));
    EXPECT_TRUE(I16(2) <= I8(8));
    EXPECT_TRUE(I8(8) > I16(3));
    EXPECT_TRUE(I16(8) > I8(8));
    EXPECT_TRUE(I8(8) >= I16(4));
    EXPECT_TRUE(I8(8) >= I16(3));
    EXPECT_TRUE(I16(8) >= I8(8));
    EXPECT_TRUE(I8(8) == I16(4));
    EXPECT_TRUE(I8(8) != I16(8));
}

UTEST(Fold, dim_mixed_stride_comparison) {
    using I8 = spio::Fold<I, 8>;
    EXPECT_TRUE(I8(8) < I(65));
    EXPECT_TRUE(I(63) < I8(8));
    EXPECT_TRUE(I8(8) <= I(64));
    EXPECT_TRUE(I8(8) <= I(65));
    EXPECT_TRUE(I(63) <= I8(8));
    EXPECT_TRUE(I8(8) > I(55));
    EXPECT_TRUE(I(70) > I8(8));
    EXPECT_TRUE(I8(8) >= I(56));
    EXPECT_TRUE(I8(8) >= I(57));
    EXPECT_TRUE(I(64) >= I8(8));
    EXPECT_TRUE(I8(8) == I(64));
    EXPECT_TRUE(I8(8) != I(65));
}

UTEST(Fold, cast) {
    using I32 = spio::Fold<I, 32>;
    using J32 = spio::Fold<J, 32>;
    EXPECT_TRUE(I32(8).cast<J>() == J32(8));
}

UTEST(Fold, dim_copy_constructor) {
    using I32 = spio::Fold<I, 32>;
    EXPECT_TRUE(I32(I(256)) == I32(8));
}

UTEST(Fold, step) {
    using I32 = spio::Fold<I, 32>;
    EXPECT_TRUE(I32::stride == I(32));
}

UTEST(Fold, data_type) {
    using I32 = spio::Fold<I, 32>;
    EXPECT_TRUE(I32::dim_type(8) == I(8));
}

UTEST(Module, methods) {
    using M4 = spio::Module<I, 4, 16>;
    EXPECT_EQ(M4(10).get(), 2);
    EXPECT_EQ(M4(50).get(), 2);
    EXPECT_TRUE(M4(10).fold<2>().get() == (10 % 4) * 16 / 2);
}

UTEST(range, step_1) {
    using I16 = spio::Fold<I, 16>;
    int count = 0;
    for (auto i16 : spio::range(I16(8))) {
        EXPECT_TRUE(i16 == I16(count++));
    }
    EXPECT_EQ(count, 8);
}

UTEST(range, step_2) {
    using I16 = spio::Fold<I, 16>;
    size_t count = 0;
    std::array<int, 4> expect = {0, 2, 4, 6};
    for (auto i16 : spio::range_with_step<2>(I16(8))) {
        EXPECT_TRUE(i16 == I16(expect[count++]));
    }
    EXPECT_EQ(count, expect.size());
}

UTEST(range, step_2_odd_end) {
    using I16 = spio::Fold<I, 16>;
    std::array<int, 5> expect = {0, 2, 4, 6, 8};
    size_t count = 0;
    for (auto i16 : spio::range_with_step<2>(I16(9))) {
        EXPECT_TRUE(i16 == I16(expect[count++]));
    }
    EXPECT_EQ(count, expect.size());
}

UTEST(range, start_1) {
    using I16 = spio::Fold<I, 16>;
    std::array<int, 7> expect = {1, 2, 3, 4, 5, 6, 7};
    size_t count = 0;
    for (auto i16 : spio::range(I16(1), I16(8))) {
        EXPECT_TRUE(i16 == I16(expect[count++]));
    }
    EXPECT_EQ(count, expect.size());
}

UTEST(range, start_1_step_2) {
    using I16 = spio::Fold<I, 16>;
    const std::array<int, 4> expect = {1, 3, 5, 7};
    size_t count = 0;
    for (auto i16 : spio::range_with_step<2>(I16(1), I16(8))) {
        EXPECT_TRUE(i16 == I16(expect[count++]));
    }
    EXPECT_EQ(count, expect.size());
}

UTEST(reverse_range, step_1) {
    using I16 = spio::Fold<I, 16>;
    std::array<int, 8> expect = {7, 6, 5, 4, 3, 2, 1, 0};
    size_t count = 0;
    for (auto i16 : spio::reverse_range(I16(8))) {
        EXPECT_EQ(i16.get(), expect[count++]);
    }
    EXPECT_EQ(count, expect.size());
}

/// One can fold the limit of the range to iterate over folded dimensions.
UTEST(range, fold_dim_range) {
    using I16 = spio::Fold<I, 16>;
    size_t count = 0;
    int n = 33;
    for (auto i16 : spio::range(I(n).fold<16>())) {
        EXPECT_TRUE(i16 == I16(count++));
    }
    EXPECT_EQ(count, n / 16);
}
