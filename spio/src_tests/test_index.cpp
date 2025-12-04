#include "dim_test_types.h"
#include "spio/compound_index.h"

using namespace spio;

UTEST(Index1D, get) {
    using Idx = CompoundIndex<DimInfo<I, 16, 1>>;
    for (unsigned offset = 0; offset < 16; ++offset) {
        Idx idx(offset);
        EXPECT_EQ(idx.get<I>(), offset);
    }
}

UTEST(Index2D, get) {
    // I x J matrix with size 32 x 8.
    using Idx = CompoundIndex<DimInfo<I, 32, 8>, DimInfo<J, 8, 1>>;
    for (unsigned offset = 0; offset < 256; ++offset) {
        Idx idx(offset);
        EXPECT_EQ(idx.get<I>(), offset / 8);
        EXPECT_EQ(idx.get<J>(), offset % 8);
    }
}

UTEST(Index3D, get) {
    // I x J x K tensor with size 16 x 8 x 4
    // Stride for K is 1
    // Stride for J is 4 (= size of K)
    // Stride for I is 32 (= size of J * stride of J)
    using Idx = CompoundIndex<DimInfo<I, 16, 32>, DimInfo<J, 8, 4>, DimInfo<K, 4, 1>>;

    for (unsigned offset = 0; offset < 16 * 8 * 4; ++offset) {
        Idx idx(offset);
        EXPECT_EQ(idx.get<I>(), offset / (8 * 4));
        EXPECT_EQ(idx.get<J>(), (offset / 4) % 8);
        EXPECT_EQ(idx.get<K>(), offset % 4);
    }
}

UTEST(IndexSize, total_size) {
    // 1D index
    using Idx1D = CompoundIndex<DimInfo<I, 16, 1>>;
    EXPECT_EQ(Idx1D::total_size, 16);
    EXPECT_EQ(Idx1D::size(), 16);

    // 2D index
    using Idx2D = CompoundIndex<DimInfo<I, 32, 8>, DimInfo<J, 8, 1>>;
    EXPECT_EQ(Idx2D::total_size, 32 * 8);
    EXPECT_EQ(Idx2D::size(), 32 * 8);

    // 3D index
    using Idx3D = CompoundIndex<DimInfo<I, 16, 32>, DimInfo<J, 8, 4>, DimInfo<K, 4, 1>>;
    EXPECT_EQ(Idx3D::total_size, 16 * 8 * 4);
    EXPECT_EQ(Idx3D::size(), 16 * 8 * 4);
}

// ============================================================================
// Cast tests - replacing base dimensions
// ============================================================================

UTEST(IndexCast, simple_dimension_replacement) {
    // Cast I to J in a simple 1D index
    using IdxI = CompoundIndex<DimInfo<I, 16, 1>>;
    using IdxJ = CompoundIndex<DimInfo<J, 16, 1>>;

    IdxI idx_i(5);
    auto idx_j = idx_i.cast<I, J>();

    static_assert(std::is_same<decltype(idx_j), IdxJ>::value, "Cast should produce IdxJ");
    EXPECT_EQ(idx_j.get<J>(), J(5));
    EXPECT_EQ(idx_j.offset(), 5);
}

UTEST(IndexCast, preserves_offset) {
    // Casting should preserve the offset value
    using IdxI = CompoundIndex<DimInfo<I, 32, 8>, DimInfo<J, 8, 1>>;

    for (unsigned offset = 0; offset < 256; ++offset) {
        IdxI idx(offset);
        auto casted = idx.cast<I, K>(); // Replace I with K
        EXPECT_EQ(casted.offset(), offset);
        EXPECT_EQ(casted.get<K>().get(), idx.get<I>().get());
        EXPECT_EQ(casted.get<J>().get(), idx.get<J>().get());
    }
}

UTEST(IndexCast, folded_dimension_preserves_fold) {
    // Cast should preserve fold structure: Fold<I, 8> -> Fold<J, 8>
    using FoldI8 = Fold<I, 8>;
    using FoldJ8 = Fold<J, 8>;
    using IdxFoldI = CompoundIndex<DimInfo<FoldI8, 4, 1>>;
    using IdxFoldJ = CompoundIndex<DimInfo<FoldJ8, 4, 1>>;

    IdxFoldI idx(3);
    auto casted = idx.cast<I, J>();

    static_assert(std::is_same<decltype(casted), IdxFoldJ>::value,
                  "Cast should preserve fold and replace base dim");
    EXPECT_EQ(casted.get<FoldJ8>().get(), idx.get<FoldI8>().get());
    EXPECT_EQ(casted.offset(), 3);
}

UTEST(IndexCast, multiple_folds_same_base) {
    // When multiple dimensions share the same base, all should be replaced
    using FoldI16 = Fold<I, 16>;
    using FoldI8 = Fold<I, 8>;
    using FoldJ16 = Fold<J, 16>;
    using FoldJ8 = Fold<J, 8>;

    using IdxI = CompoundIndex<DimInfo<FoldI16, 2, 2>, DimInfo<FoldI8, 4, 1>>;
    using IdxJ = CompoundIndex<DimInfo<FoldJ16, 2, 2>, DimInfo<FoldJ8, 4, 1>>;

    IdxI idx(5);
    auto casted = idx.cast<I, J>();

    static_assert(std::is_same<decltype(casted), IdxJ>::value,
                  "All occurrences of base dim I should be replaced with J");
    EXPECT_EQ(casted.offset(), 5);
    EXPECT_EQ(casted.get<FoldJ16>().get(), idx.get<FoldI16>().get());
    EXPECT_EQ(casted.get<FoldJ8>().get(), idx.get<FoldI8>().get());
}

UTEST(IndexCast, mixed_folded_and_unfolded) {
    // Mix of plain dimension and folded dimension with same base
    using FoldI8 = Fold<I, 8>;
    using FoldJ8 = Fold<J, 8>;

    using IdxI = CompoundIndex<DimInfo<I, 4, 4>, DimInfo<FoldI8, 2, 1>>;
    using IdxJ = CompoundIndex<DimInfo<J, 4, 4>, DimInfo<FoldJ8, 2, 1>>;

    IdxI idx(6);
    auto casted = idx.cast<I, J>();

    static_assert(std::is_same<decltype(casted), IdxJ>::value,
                  "Both I and Fold<I, 8> should be replaced");
    EXPECT_EQ(casted.offset(), 6);
    EXPECT_EQ(casted.get<J>().get(), idx.get<I>().get());
    EXPECT_EQ(casted.get<FoldJ8>().get(), idx.get<FoldI8>().get());
}

UTEST(IndexCast, unrelated_dimensions_unchanged) {
    // Dimensions with different base should not be affected
    using IdxIJ = CompoundIndex<DimInfo<I, 8, 4>, DimInfo<J, 4, 1>>;

    IdxIJ idx(15);
    auto casted = idx.cast<I, K>(); // Only replace I with K, J unchanged

    EXPECT_EQ(casted.offset(), 15);
    EXPECT_EQ(casted.get<K>().get(), idx.get<I>().get());
    EXPECT_EQ(casted.get<J>().get(), idx.get<J>().get());
}

UTEST(IndexCast, no_matching_dimension) {
    // Casting a dimension that doesn't exist should be a no-op
    using IdxIJ = CompoundIndex<DimInfo<I, 8, 4>, DimInfo<J, 4, 1>>;

    IdxIJ idx(10);
    auto casted = idx.cast<K, L>(); // K doesn't exist, nothing to replace

    static_assert(detail::is_same<decltype(casted), IdxIJ>::value,
                  "No dimensions should change if base dim not found");
    EXPECT_EQ(casted.offset(), 10);
}

UTEST(IndexCast, nested_fold) {
    // Nested folds: Fold<Fold<I, 8>, 4> should become Fold<Fold<J, 8>, 4>
    using NestedFoldI = Fold<Fold<I, 8>, 4>;
    using NestedFoldJ = Fold<Fold<J, 8>, 4>;

    using IdxI = CompoundIndex<DimInfo<NestedFoldI, 2, 1>>;
    using IdxJ = CompoundIndex<DimInfo<NestedFoldJ, 2, 1>>;

    IdxI idx(1);
    auto casted = idx.cast<I, J>();

    static_assert(std::is_same<decltype(casted), IdxJ>::value,
                  "Nested fold base dim should be replaced");
    EXPECT_EQ(casted.offset(), 1);
    EXPECT_EQ(casted.get<NestedFoldJ>().get(), idx.get<NestedFoldI>().get());
}

UTEST(IndexCast, chained_casts) {
    // Multiple casts in sequence
    using IdxI = CompoundIndex<DimInfo<I, 16, 1>>;

    IdxI idx(7);
    auto idx_j = idx.cast<I, J>();
    auto idx_k = idx_j.cast<J, K>();

    EXPECT_EQ(idx_k.get<K>(), K(7));
    EXPECT_EQ(idx_k.offset(), 7);
}
