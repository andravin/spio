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
    EXPECT_TRUE(I32::stride == 32);
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

UTEST(Module, properties) {
    using M4 = spio::Module<I, 4, 16>;
    EXPECT_EQ(M4::size, 4);
    EXPECT_EQ(M4::stride, 16);
    EXPECT_TRUE((std::is_same<M4::dim_type, I>::value));
}

UTEST(Module, unfold) {
    using M4 = spio::Module<I, 4, 16>;
    EXPECT_EQ(M4(2).unfold().get(), 2 * 16);
    EXPECT_EQ(M4(0).unfold().get(), 0);
    EXPECT_EQ(M4(3).unfold().get(), 3 * 16);
}

UTEST(Module, cast) {
    using MI4 = spio::Module<I, 4, 16>;
    using MJ4 = spio::Module<J, 4, 16>;
    EXPECT_TRUE(MI4(2).cast<J>() == MJ4(2));
    EXPECT_EQ(MI4(3).cast<J>().get(), 3);
}

UTEST(Module, construct_from_dim) {
    using M4 = spio::Module<I, 4, 16>;
    // Module(DimType dim) constructor: (dim.get() / Stride) % Size
    EXPECT_EQ(M4(I(32)).get(), (32 / 16) % 4); // = 2
    EXPECT_EQ(M4(I(64)).get(), (64 / 16) % 4); // = 0
    EXPECT_EQ(M4(I(80)).get(), (80 / 16) % 4); // = 1
}

UTEST(Module, arithmetic) {
    using M4 = spio::Module<I, 4, 16>;
    EXPECT_TRUE(M4(1) + M4(2) == M4(3));
    EXPECT_TRUE(M4(3) - M4(1) == M4(2));
    EXPECT_TRUE(M4(1) - M4(2) == M4(-1));
}

UTEST(Module, comparison) {
    using M4 = spio::Module<I, 4, 16>;
    EXPECT_TRUE(M4(1) < M4(2));
    EXPECT_TRUE(M4(2) > M4(1));
    EXPECT_TRUE(M4(1) <= M4(2));
    EXPECT_TRUE(M4(2) >= M4(1));
    EXPECT_TRUE(M4(2) == M4(2));
    EXPECT_TRUE(M4(2) <= M4(2));
    EXPECT_TRUE(M4(2) >= M4(2));
    EXPECT_TRUE(M4(1) != M4(2));
}

UTEST(Module, dim_addition) {
    using M4 = spio::Module<I, 4, 16>;
    // Module unfolds to base dim for cross-type arithmetic
    EXPECT_TRUE(M4(2) + I(8) == I(40)); // 2*16 + 8 = 40
    EXPECT_TRUE(I(8) + M4(2) == I(40));
}

UTEST(Module, dim_subtraction) {
    using M4 = spio::Module<I, 4, 16>;
    EXPECT_TRUE(M4(2) - I(8) == I(24)); // 2*16 - 8 = 24
    EXPECT_TRUE(I(40) - M4(2) == I(8)); // 40 - 2*16 = 8
}

UTEST(Module, fold_addition) {
    using M4 = spio::Module<I, 4, 16>;
    using I8 = spio::Fold<I, 8>;
    // M4(2) unfolds to 32, I8(2) unfolds to 16, result in finer stride (8)
    EXPECT_TRUE(M4(2) + I8(2) == I8(6)); // 32/8 + 2 = 6
    EXPECT_TRUE(I8(2) + M4(2) == I8(6));
}

UTEST(Module, fold_subtraction) {
    using M4 = spio::Module<I, 4, 16>;
    using I8 = spio::Fold<I, 8>;
    EXPECT_TRUE(M4(2) - I8(2) == I8(2)); // 32/8 - 2 = 2
    EXPECT_TRUE(I8(6) - M4(2) == I8(2)); // 6 - 32/8 = 2
}

UTEST(Module, dim_comparison) {
    using M4 = spio::Module<I, 4, 16>;
    EXPECT_TRUE(M4(2) == I(32)); // 2*16 = 32
    EXPECT_TRUE(M4(2) < I(33));
    EXPECT_TRUE(M4(2) > I(31));
    EXPECT_TRUE(M4(2) <= I(32));
    EXPECT_TRUE(M4(2) >= I(32));
    EXPECT_TRUE(M4(2) != I(33));
}

UTEST(Module, fold_comparison) {
    using M4 = spio::Module<I, 4, 16>;
    using I8 = spio::Fold<I, 8>;
    EXPECT_TRUE(M4(2) == I8(4)); // 2*16 = 4*8
    EXPECT_TRUE(M4(2) < I8(5));
    EXPECT_TRUE(M4(2) > I8(3));
    EXPECT_TRUE(M4(2) <= I8(4));
    EXPECT_TRUE(M4(2) >= I8(4));
    EXPECT_TRUE(M4(2) != I8(5));
}

UTEST(Dim, unfold) {
    // Dim::unfold() returns itself (stride is 1)
    EXPECT_TRUE(I(7).unfold() == I(7));
    EXPECT_EQ(I(42).unfold().get(), 42);
}

UTEST(Dim, stride_property) {
    EXPECT_EQ(I::stride, 1);
    EXPECT_EQ(J::stride, 1);
}

UTEST(Dim, module_method) {
    // Dim::module<Size>() returns Module<Derived, Size, 1>
    auto m = I(10).module<4>();
    EXPECT_EQ(m.get(), 10 % 4);     // = 2
    EXPECT_EQ(m.unfold().get(), 2); // stride is 1
    EXPECT_EQ(decltype(m)::size, 4);
    EXPECT_EQ(decltype(m)::stride, 1);

    // Test with different values
    EXPECT_EQ(I(0).module<4>().get(), 0);
    EXPECT_EQ(I(3).module<4>().get(), 3);
    EXPECT_EQ(I(4).module<4>().get(), 0);
    EXPECT_EQ(I(7).module<4>().get(), 3);
    EXPECT_EQ(I(100).module<8>().get(), 100 % 8);
}

UTEST(Fold, module_method) {
    using I8 = spio::Fold<I, 8>;
    // Fold::module<Size>() returns Module<DimType, Size, Stride>
    auto m = I8(10).module<4>();
    EXPECT_EQ(m.get(), 10 % 4);         // = 2
    EXPECT_EQ(m.unfold().get(), 2 * 8); // stride is 8, so unfold = 16
    EXPECT_EQ(decltype(m)::size, 4);
    EXPECT_EQ(decltype(m)::stride, 8);

    // Test with different values
    EXPECT_EQ(I8(0).module<4>().get(), 0);
    EXPECT_EQ(I8(3).module<4>().get(), 3);
    EXPECT_EQ(I8(4).module<4>().get(), 0);
    EXPECT_EQ(I8(7).module<4>().get(), 3);
    EXPECT_EQ(I8(100).module<8>().get(), 100 % 8);
}

UTEST(Fold, module_preserves_stride) {
    using I16 = spio::Fold<I, 16>;
    using I32 = spio::Fold<I, 32>;

    // Module from Fold<I, 16> should have stride 16
    auto m16 = I16(5).module<4>();
    EXPECT_EQ(decltype(m16)::stride, 16);
    EXPECT_EQ(m16.unfold().get(), (5 % 4) * 16); // = 16

    // Module from Fold<I, 32> should have stride 32
    auto m32 = I32(7).module<4>();
    EXPECT_EQ(decltype(m32)::stride, 32);
    EXPECT_EQ(m32.unfold().get(), (7 % 4) * 32); // = 96
}

UTEST(min_max, same_type) {
    EXPECT_TRUE(spio::min(I(5), I(10)) == I(5));
    EXPECT_TRUE(spio::max(I(5), I(10)) == I(10));
    EXPECT_TRUE(spio::min(I(10), I(5)) == I(5));
    EXPECT_TRUE(spio::max(I(10), I(5)) == I(10));
}

UTEST(min_max, fold_same_stride) {
    using I8 = spio::Fold<I, 8>;
    EXPECT_TRUE(spio::min(I8(5), I8(10)) == I8(5));
    EXPECT_TRUE(spio::max(I8(5), I8(10)) == I8(10));
}

UTEST(min_max, module_same_type) {
    using M4 = spio::Module<I, 4, 16>;
    EXPECT_TRUE(spio::min(M4(1), M4(3)) == M4(1));
    EXPECT_TRUE(spio::max(M4(1), M4(3)) == M4(3));
}

UTEST(min_max, fold_different_stride) {
    using I8 = spio::Fold<I, 8>;
    using I16 = spio::Fold<I, 16>;
    // I8(8) = 64, I16(5) = 80, result in finer stride (8)
    EXPECT_TRUE(spio::min(I8(8), I16(5)) == I8(8));
    EXPECT_TRUE(spio::max(I8(8), I16(5)) == I8(10)); // 80/8 = 10
}

UTEST(min_max, dim_and_fold) {
    using I8 = spio::Fold<I, 8>;
    // I8(8) = 64, I(50) = 50
    EXPECT_TRUE(spio::min(I8(8), I(50)) == I(50));
    EXPECT_TRUE(spio::max(I8(8), I(50)) == I(64));
    EXPECT_TRUE(spio::min(I(50), I8(8)) == I(50));
    EXPECT_TRUE(spio::max(I(50), I8(8)) == I(64));
}

UTEST(min_max, module_and_dim) {
    using M4 = spio::Module<I, 4, 16>;
    // M4(2) unfolds to 32
    EXPECT_TRUE(spio::min(M4(2), I(50)) == I(32));
    EXPECT_TRUE(spio::max(M4(2), I(50)) == I(50));
    EXPECT_TRUE(spio::min(I(50), M4(2)) == I(32));
    EXPECT_TRUE(spio::max(I(50), M4(2)) == I(50));
}

UTEST(min_max, module_and_fold) {
    using M4 = spio::Module<I, 4, 16>;
    using I8 = spio::Fold<I, 8>;
    // M4(2) = 32, I8(5) = 40, result in stride 8
    EXPECT_TRUE(spio::min(M4(2), I8(5)) == I8(4)); // 32/8 = 4
    EXPECT_TRUE(spio::max(M4(2), I8(5)) == I8(5));
}

UTEST(cross_type, negative_values) {
    using I8 = spio::Fold<I, 8>;
    EXPECT_TRUE(I8(2) - I(20) == I(-4)); // 16 - 20 = -4
    EXPECT_TRUE(I(10) - I8(2) == I(-6)); // 10 - 16 = -6
    EXPECT_TRUE(I8(-1) + I(16) == I(8)); // -8 + 16 = 8
}

UTEST(cross_type, module_negative) {
    using M4 = spio::Module<I, 4, 16>;
    EXPECT_TRUE(M4(1) - I(20) == I(-4)); // 16 - 20 = -4
    EXPECT_TRUE(I(10) - M4(1) == I(-6)); // 10 - 16 = -6
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
