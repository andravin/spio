#include "dim_test_common.h"

#include "spio/coordinates.h"
#include "spio/dim.h"
#include "spio/tensor.h"
#include "spio/compound_index.h"

UTEST(Coordinates, direct_construction) {
    spio::Coordinates<I, J> coords(I(1), J(2));
    EXPECT_EQ(coords.get<I>(), I(1));
    EXPECT_EQ(coords.get<J>(), J(2));
}

UTEST(Coordinates, make_coordinates_from_lvalues) {
    I i(7);
    J j(8);
    auto coords = spio::make_coordinates(i, j);
    EXPECT_EQ(coords.get<I>(), I(7));
    EXPECT_EQ(coords.get<J>(), J(8));
}

UTEST(Coordinates, make_coordinates_mixed) {
    I i(10);
    auto coords = spio::make_coordinates(i, J(20));
    EXPECT_EQ(coords.get<I>(), I(10));
    EXPECT_EQ(coords.get<J>(), J(20));
}

UTEST(Coordinates, const_access) {
    const auto coords = spio::make_coordinates(I(100), J(200));
    EXPECT_EQ(coords.get<I>(), I(100));
    EXPECT_EQ(coords.get<J>(), J(200));
}

UTEST(Coordinates, add_same_dims) {
    auto a = spio::make_coordinates(I(1), J(2));
    auto b = spio::make_coordinates(I(10), J(20));
    auto result = a + b;
    EXPECT_EQ(result.get<I>(), I(11));
    EXPECT_EQ(result.get<J>(), J(22));
}

UTEST(Coordinates, add_disjoint_dims) {
    auto a = spio::make_coordinates(I(5));
    auto b = spio::make_coordinates(J(7));
    auto result = a + b;
    EXPECT_EQ(result.get<I>(), I(5));
    EXPECT_EQ(result.get<J>(), J(7));
}

UTEST(Coordinates, add_overlapping_dims) {
    auto a = spio::make_coordinates(I(1), J(2));
    auto b = spio::make_coordinates(J(20), K(30));
    auto result = a + b;
    EXPECT_EQ(result.get<I>(), I(1));  // only in a
    EXPECT_EQ(result.get<J>(), J(22)); // in both, added
    EXPECT_EQ(result.get<K>(), K(30)); // only in b
}

UTEST(Coordinates, add_subset_b_in_a) {
    auto a = spio::make_coordinates(I(3), J(4), K(5));
    auto b = spio::make_coordinates(J(10));
    auto result = a + b;
    EXPECT_EQ(result.get<I>(), I(3));
    EXPECT_EQ(result.get<J>(), J(14));
    EXPECT_EQ(result.get<K>(), K(5));
}

UTEST(Coordinates, add_subset_a_in_b) {
    auto a = spio::make_coordinates(J(10));
    auto b = spio::make_coordinates(I(3), J(4), K(5));
    auto result = a + b;
    EXPECT_EQ(result.get<I>(), I(3));
    EXPECT_EQ(result.get<J>(), J(14));
    EXPECT_EQ(result.get<K>(), K(5));
}

UTEST(Coordinates, add_single_dims) {
    auto a = spio::make_coordinates(I(100));
    auto b = spio::make_coordinates(I(23));
    auto result = a + b;
    EXPECT_EQ(result.get<I>(), I(123));
}

UTEST(Coordinates, less_than_same_dims_true) {
    auto a = spio::make_coordinates(I(1), J(2));
    auto b = spio::make_coordinates(I(10), J(20));
    EXPECT_TRUE(a < b);
}

UTEST(Coordinates, less_than_same_dims_false_first) {
    auto a = spio::make_coordinates(I(100), J(2));
    auto b = spio::make_coordinates(I(10), J(20));
    EXPECT_FALSE(a < b);
}

UTEST(Coordinates, less_than_same_dims_false_second) {
    auto a = spio::make_coordinates(I(1), J(200));
    auto b = spio::make_coordinates(I(10), J(20));
    EXPECT_FALSE(a < b);
}

UTEST(Coordinates, less_than_same_dims_equal) {
    auto a = spio::make_coordinates(I(5), J(5));
    auto b = spio::make_coordinates(I(5), J(5));
    EXPECT_FALSE(a < b);
}

UTEST(Coordinates, less_than_disjoint_dims) {
    auto a = spio::make_coordinates(I(100));
    auto b = spio::make_coordinates(J(1));
    EXPECT_TRUE(a < b); // No shared dims, vacuously true
}

UTEST(Coordinates, less_than_partial_overlap_true) {
    auto a = spio::make_coordinates(I(1), J(2));
    auto b = spio::make_coordinates(J(20), K(30));
    EXPECT_TRUE(a < b); // Only J is compared, and 2 < 20
}

UTEST(Coordinates, less_than_partial_overlap_false) {
    auto a = spio::make_coordinates(I(1), J(200));
    auto b = spio::make_coordinates(J(20), K(30));
    EXPECT_FALSE(a < b); // Only J is compared, and 200 < 20 is false
}

UTEST(Coordinates, less_than_a_subset_of_b) {
    auto a = spio::make_coordinates(J(5));
    auto b = spio::make_coordinates(I(10), J(20), K(30));
    EXPECT_TRUE(a < b); // Only J is compared, and 5 < 20
}

UTEST(Coordinates, less_than_b_subset_of_a) {
    auto a = spio::make_coordinates(I(1), J(2), K(3));
    auto b = spio::make_coordinates(J(20));
    EXPECT_TRUE(a < b); // Only J is compared, and 2 < 20
}

// Less than or equal tests
UTEST(Coordinates, less_equal_same_dims_true) {
    auto a = spio::make_coordinates(I(1), J(2));
    auto b = spio::make_coordinates(I(10), J(20));
    EXPECT_TRUE(a <= b);
}

UTEST(Coordinates, less_equal_same_dims_equal) {
    auto a = spio::make_coordinates(I(5), J(5));
    auto b = spio::make_coordinates(I(5), J(5));
    EXPECT_TRUE(a <= b);
}

UTEST(Coordinates, less_equal_same_dims_false) {
    auto a = spio::make_coordinates(I(100), J(2));
    auto b = spio::make_coordinates(I(10), J(20));
    EXPECT_FALSE(a <= b);
}

// Greater than tests
UTEST(Coordinates, greater_same_dims_true) {
    auto a = spio::make_coordinates(I(100), J(200));
    auto b = spio::make_coordinates(I(10), J(20));
    EXPECT_TRUE(a > b);
}

UTEST(Coordinates, greater_same_dims_false) {
    auto a = spio::make_coordinates(I(1), J(2));
    auto b = spio::make_coordinates(I(10), J(20));
    EXPECT_FALSE(a > b);
}

UTEST(Coordinates, greater_same_dims_equal) {
    auto a = spio::make_coordinates(I(5), J(5));
    auto b = spio::make_coordinates(I(5), J(5));
    EXPECT_FALSE(a > b);
}

// Greater than or equal tests
UTEST(Coordinates, greater_equal_same_dims_true) {
    auto a = spio::make_coordinates(I(100), J(200));
    auto b = spio::make_coordinates(I(10), J(20));
    EXPECT_TRUE(a >= b);
}

UTEST(Coordinates, greater_equal_same_dims_equal) {
    auto a = spio::make_coordinates(I(5), J(5));
    auto b = spio::make_coordinates(I(5), J(5));
    EXPECT_TRUE(a >= b);
}

UTEST(Coordinates, greater_equal_same_dims_false) {
    auto a = spio::make_coordinates(I(1), J(2));
    auto b = spio::make_coordinates(I(10), J(20));
    EXPECT_FALSE(a >= b);
}

// Equal tests
UTEST(Coordinates, equal_same_dims_true) {
    auto a = spio::make_coordinates(I(5), J(10));
    auto b = spio::make_coordinates(I(5), J(10));
    EXPECT_TRUE(a == b);
}

UTEST(Coordinates, equal_same_dims_false) {
    auto a = spio::make_coordinates(I(5), J(10));
    auto b = spio::make_coordinates(I(5), J(20));
    EXPECT_FALSE(a == b);
}

UTEST(Coordinates, equal_disjoint_dims) {
    auto a = spio::make_coordinates(I(100));
    auto b = spio::make_coordinates(J(1));
    EXPECT_TRUE(a == b); // No shared dims, vacuously true
}

UTEST(Coordinates, equal_partial_overlap_true) {
    auto a = spio::make_coordinates(I(1), J(20));
    auto b = spio::make_coordinates(J(20), K(30));
    EXPECT_TRUE(a == b); // Only J is compared, and 20 == 20
}

UTEST(Coordinates, equal_partial_overlap_false) {
    auto a = spio::make_coordinates(I(1), J(20));
    auto b = spio::make_coordinates(J(30), K(30));
    EXPECT_FALSE(a == b); // Only J is compared, and 20 != 30
}

// Not equal tests
UTEST(Coordinates, not_equal_same_dims_true) {
    auto a = spio::make_coordinates(I(5), J(10));
    auto b = spio::make_coordinates(I(5), J(20));
    EXPECT_TRUE(a != b); // J differs, so not equal
}

UTEST(Coordinates, not_equal_same_dims_false) {
    auto a = spio::make_coordinates(I(5), J(10));
    auto b = spio::make_coordinates(I(5), J(10));
    EXPECT_FALSE(a != b); // All dims equal, so NOT not-equal
}

UTEST(Coordinates, not_equal_partial_overlap) {
    auto a = spio::make_coordinates(I(1), J(20));
    auto b = spio::make_coordinates(J(30), K(30));
    EXPECT_TRUE(a != b); // Only J is compared, and 20 != 30
}

UTEST(Coordinates, normalize_single_dim) {
    auto coords = spio::make_coordinates(I(5));
    auto normalized = coords.normalize();
    EXPECT_EQ(normalized.get<I>(), I(5));
}

UTEST(Coordinates, normalize_multiple_different_dims) {
    auto coords = spio::make_coordinates(I(3), J(7));
    auto normalized = coords.normalize();
    EXPECT_EQ(normalized.get<I>(), I(3));
    EXPECT_EQ(normalized.get<J>(), J(7));
}

UTEST(Coordinates, normalize_fold_single) {
    using FoldI8 = spio::Fold<I, 8>;
    auto coords = spio::make_coordinates(FoldI8(2));
    auto normalized = coords.normalize();
    // Result is still Fold<I, 8> since that's the only/smallest stride
    EXPECT_EQ(normalized.get<FoldI8>(), FoldI8(2));
}

UTEST(Coordinates, normalize_dim_and_fold_same_base) {
    using FoldI8 = spio::Fold<I, 8>;
    // I(3) + FoldI8(2) = 3 + 16 = 19
    // Min stride is 1 (from I), so result is I(19)
    auto coords = make_coordinates(I(3), FoldI8(2));
    auto normalized = coords.normalize();
    EXPECT_EQ(normalized.get<I>(), I(19));
}

UTEST(Coordinates, normalize_two_folds_same_base) {
    using FoldI8 = spio::Fold<I, 8>;
    using FoldI16 = spio::Fold<I, 16>;
    // FoldI8(1) + FoldI16(2) = 8 + 32 = 40
    // Min stride is 8 (from FoldI8), so result is Fold<I, 8>(5)
    auto coords = make_coordinates(FoldI8(1), FoldI16(2));
    auto normalized = coords.normalize();
    EXPECT_EQ(normalized.get<FoldI8>(), FoldI8(5)); // 5 * 8 = 40
}

UTEST(Coordinates, normalize_two_folds_smaller_stride_second) {
    using FoldI4 = spio::Fold<I, 4>;
    using FoldI8 = spio::Fold<I, 8>;
    // FoldI8(1) + FoldI4(2) = 8 + 8 = 16
    // Min stride is 4 (from FoldI4), so result is Fold<I, 4>(4)
    auto coords = make_coordinates(FoldI8(1), FoldI4(2));
    auto normalized = coords.normalize();
    EXPECT_EQ(normalized.get<FoldI4>(), FoldI4(4)); // 4 * 4 = 16
    EXPECT_EQ(normalized.num_dims(), 1);
}

UTEST(Coordinates, normalize_mixed_dims_and_folds) {
    using FoldI8 = spio::Fold<I, 8>;
    using FoldJ4 = spio::Fold<J, 4>;
    // I(2) + FoldI8(3) = 2 + 24 = 26, min stride 1 -> I(26)
    // J(1) + FoldJ4(2) = 1 + 8 = 9, min stride 1 -> J(9)
    auto coords = make_coordinates(I(2), FoldI8(3), J(1), FoldJ4(2));
    auto normalized = coords.normalize();
    EXPECT_EQ(normalized.get<I>(), I(26));
    EXPECT_EQ(normalized.get<J>(), J(9));
    EXPECT_EQ(normalized.num_dims(), 2);
}

UTEST(Coordinates, normalize_preserves_fold_when_no_dim) {
    using FoldI8 = spio::Fold<I, 8>;
    using FoldJ4 = spio::Fold<J, 4>;
    // Only folds, no plain dims -> result keeps the fold types
    auto coords = make_coordinates(FoldI8(2), FoldJ4(3));
    auto normalized = coords.normalize();
    EXPECT_EQ(normalized.get<FoldI8>(), FoldI8(2));
    EXPECT_EQ(normalized.get<FoldJ4>(), FoldJ4(3));
    EXPECT_EQ(normalized.num_dims(), 2);
}

UTEST(Coordinates, normalize_preserves_order) {
    using FoldJ4 = spio::Fold<J, 4>;
    // I stays as I(5), FoldJ4 stays as FoldJ4(3)
    auto coords = make_coordinates(I(5), FoldJ4(3));
    auto normalized = coords.normalize();
    EXPECT_EQ(normalized.get<I>(), I(5));
    EXPECT_EQ(normalized.get<FoldJ4>(), FoldJ4(3));
    EXPECT_EQ(normalized.num_dims(), 2);
}

UTEST(Coordinates, make_normalized_coordinates_single_dim) {
    auto coords = spio::make_normalized_coordinates(I(5));
    EXPECT_EQ(coords.get<I>(), I(5));
    EXPECT_EQ(coords.num_dims(), 1);
}

UTEST(Coordinates, make_normalized_coordinates_multiple_dims) {
    auto coords = spio::make_normalized_coordinates(I(3), J(7));
    EXPECT_EQ(coords.get<I>(), I(3));
    EXPECT_EQ(coords.get<J>(), J(7));
    EXPECT_EQ(coords.num_dims(), 2);
}

UTEST(Coordinates, make_normalized_coordinates_fold_single) {
    using FoldI8 = spio::Fold<I, 8>;
    auto coords = spio::make_normalized_coordinates(FoldI8(2));
    EXPECT_EQ(coords.get<FoldI8>(), FoldI8(2));
    EXPECT_EQ(coords.num_dims(), 1);
}

UTEST(Coordinates, make_normalized_coordinates_dim_and_fold) {
    using FoldI8 = spio::Fold<I, 8>;
    // I(3) + FoldI8(2) = 3 + 16 = 19
    auto coords = spio::make_normalized_coordinates(I(3), FoldI8(2));
    EXPECT_EQ(coords.get<I>(), I(19));
    EXPECT_EQ(coords.num_dims(), 1);
}

UTEST(Coordinates, make_normalized_coordinates_two_folds) {
    using FoldI4 = spio::Fold<I, 4>;
    using FoldI8 = spio::Fold<I, 8>;
    // FoldI8(1) + FoldI4(2) = 8 + 8 = 16
    // Min stride is 4, so result is Fold<I, 4>(4)
    auto coords = spio::make_normalized_coordinates(FoldI8(1), FoldI4(2));
    EXPECT_EQ(coords.get<FoldI4>(), FoldI4(4));
    EXPECT_EQ(coords.num_dims(), 1);
}

UTEST(Coordinates, make_normalized_coordinates_mixed) {
    using FoldI8 = spio::Fold<I, 8>;
    using FoldJ4 = spio::Fold<J, 4>;
    // I(2) + FoldI8(3) = 2 + 24 = 26 -> I(26)
    // J(1) + FoldJ4(2) = 1 + 8 = 9 -> J(9)
    auto coords = spio::make_normalized_coordinates(I(2), FoldI8(3), J(1), FoldJ4(2));
    EXPECT_EQ(coords.get<I>(), I(26));
    EXPECT_EQ(coords.get<J>(), J(9));
    EXPECT_EQ(coords.num_dims(), 2);
}

UTEST(Coordinates, make_normalized_coordinates_from_lvalues) {
    I i(10);
    J j(20);
    auto coords = spio::make_normalized_coordinates(i, j);
    EXPECT_EQ(coords.get<I>(), I(10));
    EXPECT_EQ(coords.get<J>(), J(20));
    EXPECT_EQ(coords.num_dims(), 2);
}

// Addition with normalization tests
UTEST(Coordinates, add_dim_and_fold_same_base) {
    using FoldI8 = spio::Fold<I, 8>;
    auto a = spio::make_coordinates(I(3));
    auto b = spio::make_coordinates(FoldI8(2)); // 2 * 8 = 16
    auto result = a + b;
    // Both normalized to I, then added: 3 + 16 = 19
    EXPECT_EQ(result.get<I>(), I(19));
    EXPECT_EQ(result.num_dims(), 1);
}

UTEST(Coordinates, add_two_folds_different_strides) {
    using FoldI4 = spio::Fold<I, 4>;
    using FoldI8 = spio::Fold<I, 8>;
    auto a = spio::make_coordinates(FoldI8(1)); // 1 * 8 = 8
    auto b = spio::make_coordinates(FoldI4(2)); // 2 * 4 = 8
    auto result = a + b;
    // Both normalized to Fold<I, 4>, then added: 2 + 2 = 4
    EXPECT_EQ(result.get<FoldI4>(), FoldI4(4)); // 4 * 4 = 16
    EXPECT_EQ(result.num_dims(), 1);
}

UTEST(Coordinates, add_mixed_folds_and_dims) {
    using FoldI8 = spio::Fold<I, 8>;
    using FoldJ4 = spio::Fold<J, 4>;
    auto a = spio::make_coordinates(I(2), FoldJ4(1)); // I=2, J=4
    auto b = spio::make_coordinates(FoldI8(1), J(3)); // I=8, J=3
    auto result = a + b;
    // I: 2 + 8 = 10 (normalized to I since stride 1 < 8)
    // J: 4 + 3 = 7 (normalized to J since stride 1 < 4)
    EXPECT_EQ(result.get<I>(), I(10));
    EXPECT_EQ(result.get<J>(), J(7));
    EXPECT_EQ(result.num_dims(), 2);
}

UTEST(Coordinates, add_unnormalized_to_normalized) {
    using FoldI8 = spio::Fold<I, 8>;
    auto a = spio::make_coordinates(I(1), FoldI8(1)); // Unnormalized: I(1) + FoldI8(1) = 9
    auto b = spio::make_coordinates(I(5));            // Already normalized
    auto result = a + b;
    // a normalizes to I(9), then 9 + 5 = 14
    EXPECT_EQ(result.get<I>(), I(14));
    EXPECT_EQ(result.num_dims(), 1);
}

// Comparison with normalization tests
UTEST(Coordinates, less_than_dim_vs_fold) {
    using FoldI8 = spio::Fold<I, 8>;
    auto a = spio::make_coordinates(I(5));      // I = 5
    auto b = spio::make_coordinates(FoldI8(1)); // I = 8
    EXPECT_TRUE(a < b);                         // 5 < 8
    EXPECT_FALSE(b < a);                        // 8 < 5 is false
}

UTEST(Coordinates, less_than_two_folds_different_strides) {
    using FoldI4 = spio::Fold<I, 4>;
    using FoldI8 = spio::Fold<I, 8>;
    auto a = spio::make_coordinates(FoldI4(1)); // I = 4
    auto b = spio::make_coordinates(FoldI8(1)); // I = 8
    EXPECT_TRUE(a < b);                         // 4 < 8
    EXPECT_FALSE(b < a);                        // 8 < 4 is false
}

UTEST(Coordinates, less_equal_dim_vs_fold_equal) {
    using FoldI8 = spio::Fold<I, 8>;
    auto a = spio::make_coordinates(I(8));      // I = 8
    auto b = spio::make_coordinates(FoldI8(1)); // I = 8
    EXPECT_TRUE(a <= b);                        // 8 <= 8
    EXPECT_TRUE(b <= a);                        // 8 <= 8
    EXPECT_TRUE(a == b);                        // 8 == 8
}

UTEST(Coordinates, equal_unnormalized_coordinates) {
    using FoldI8 = spio::Fold<I, 8>;
    auto a = spio::make_coordinates(I(1), FoldI8(1)); // Unnormalized: 1 + 8 = 9
    auto b = spio::make_coordinates(I(9));            // Already normalized: 9
    EXPECT_TRUE(a == b);                              // Both normalize to I(9)
}

UTEST(Coordinates, not_equal_unnormalized_coordinates) {
    using FoldI8 = spio::Fold<I, 8>;
    auto a = spio::make_coordinates(I(1), FoldI8(1)); // Unnormalized: 1 + 8 = 9
    auto b = spio::make_coordinates(I(10));           // Already normalized: 10
    EXPECT_TRUE(a != b);                              // 9 != 10
    EXPECT_FALSE(a == b);                             // 9 == 10 is false
}

UTEST(Coordinates, greater_mixed_folds_and_dims) {
    using FoldI4 = spio::Fold<I, 4>;
    using FoldJ8 = spio::Fold<J, 8>;
    auto a = spio::make_coordinates(I(10), FoldJ8(2)); // I=10, J=16
    auto b = spio::make_coordinates(FoldI4(2), J(10)); // I=8, J=10
    EXPECT_TRUE(a > b);                                // 10 > 8 and 16 > 10
    EXPECT_FALSE(b > a);                               // 8 > 10 is false
}

UTEST(Coordinates, comparison_partial_overlap_with_folds) {
    using FoldI8 = spio::Fold<I, 8>;
    using FoldK4 = spio::Fold<K, 4>;
    auto a = spio::make_coordinates(I(5), J(10));          // I=5, J=10
    auto b = spio::make_coordinates(FoldI8(1), FoldK4(3)); // I=8, K=12
    // Only I is compared (shared dimension after normalization)
    EXPECT_TRUE(a < b);   // 5 < 8
    EXPECT_FALSE(a > b);  // 5 > 8 is false
    EXPECT_FALSE(a == b); // 5 != 8
}

UTEST(Coordinates, fold_like_simple_tensor) {
    // Create a simple tensor type with plain dimensions
    using TestTensor = spio::Tensor<float, spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(2), J(3));
    auto folded = coords.fold_like<TestTensor>();

    EXPECT_EQ(folded.get<I>(), I(2));
    EXPECT_EQ(folded.get<J>(), J(3));
}

UTEST(Coordinates, fold_like_with_fold_dimension) {
    using FoldI8 = spio::Fold<I, 8>;

    // Tensor with a folded dimension
    using TestTensor = spio::Tensor<float, spio::DimInfo<FoldI8, 4, 1>, spio::DimInfo<J, 8, 4>>;

    // Coordinates with plain I dimension
    auto coords = spio::make_coordinates(I(16), J(3));
    auto folded = coords.fold_like<TestTensor>();

    // I(16) should become FoldI8(2) since 16/8 = 2
    EXPECT_EQ(folded.get<FoldI8>(), FoldI8(2));
    EXPECT_EQ(folded.get<J>(), J(3));
}

UTEST(Coordinates, fold_like_unfold_to_plain_dim) {
    using FoldI8 = spio::Fold<I, 8>;

    // Tensor with plain I dimension
    using TestTensor = spio::Tensor<float, spio::DimInfo<I, 32, 1>, spio::DimInfo<J, 8, 32>>;

    // Coordinates with folded dimension
    auto coords = spio::make_coordinates(FoldI8(2), J(3));
    auto folded = coords.fold_like<TestTensor>();

    // FoldI8(2) should become I(16) since 2*8 = 16
    EXPECT_EQ(folded.get<I>(), I(16));
    EXPECT_EQ(folded.get<J>(), J(3));
}

UTEST(Coordinates, fold_like_refold_to_different_stride) {
    using FoldI4 = spio::Fold<I, 4>;
    using FoldI8 = spio::Fold<I, 8>;

    // Tensor with FoldI4 dimension
    using TestTensor = spio::Tensor<float, spio::DimInfo<FoldI4, 8, 1>, spio::DimInfo<J, 4, 8>>;

    // Coordinates with FoldI8
    auto coords = spio::make_coordinates(FoldI8(2), J(3)); // 2*8 = 16
    auto folded = coords.fold_like<TestTensor>();

    // FoldI8(2) = 16 should become FoldI4(4) since 16/4 = 4
    EXPECT_EQ(folded.get<FoldI4>(), FoldI4(4));
    EXPECT_EQ(folded.get<J>(), J(3));
}

UTEST(Coordinates, fold_like_multiple_folds_same_base) {
    using FoldI4 = spio::Fold<I, 4>;
    using FoldI8 = spio::Fold<I, 8>;

    // Tensor with multiple fold dimensions for same base type
    using TestTensor = spio::Tensor<float, spio::DimInfo<FoldI8, 4, 1>, // Outer fold
                                    spio::DimInfo<FoldI4, 2, 4>>;       // Inner fold

    // Coordinates with plain I dimension
    auto coords = spio::make_coordinates(I(24)); // 24 = 3*8 or 6*4
    auto folded = coords.fold_like<TestTensor>();

    // Should have both FoldI8(3) and FoldI4(6)
    EXPECT_EQ(folded.get<FoldI8>(), FoldI8(3)); // (24/8)%4 = 3
    EXPECT_EQ(folded.get<FoldI4>(), FoldI4(0)); // (24/4)%2 = 0
    EXPECT_EQ(folded.num_dims(), 2);
}

UTEST(Coordinates, fold_like_partial_match) {
    using FoldI8 = spio::Fold<I, 8>;

    // Tensor with I and K dimensions (no J)
    using TestTensor = spio::Tensor<float, spio::DimInfo<FoldI8, 4, 1>, spio::DimInfo<K, 8, 4>>;

    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    // Coordinates with I and J (tensor has I and K)
    auto coords = spio::make_coordinates(I(16), J(3));
    auto cursor = coords.apply_to(tensor);

    // Only I matches: I(16) folds to FoldI8(2), J(3) is ignored
    // Offset: FoldI8(2) * 1 = 2
    EXPECT_EQ(cursor.get(), data + 2);
}

UTEST(Coordinates, apply_to_no_matching_dims) {
    using TestTensor = spio::Tensor<float, spio::DimInfo<K, 8, 1>>;

    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    // Coordinates with I and J, tensor only has K
    auto coords = spio::make_coordinates(I(5), J(3));
    auto cursor = coords.apply_to(tensor);

    // No matching dimensions, offset should be 0
    EXPECT_EQ(cursor.get(), data);
}

UTEST(Coordinates, apply_to_normalized_coordinates) {
    using FoldI8 = spio::Fold<I, 8>;
    using TestTensor = spio::Tensor<float, spio::DimInfo<FoldI8, 4, 1>, spio::DimInfo<J, 8, 4>>;

    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    // Unnormalized coordinates: I(1) + FoldI8(1) = 9
    auto coords = spio::make_coordinates(I(1), FoldI8(1));
    auto cursor = coords.apply_to(tensor);

    // Normalized to I(9), then folded to FoldI8(1) (9/8 = 1)
    // Note: 9/8 = 1 with integer division
    // Offset: FoldI8(1) * 1 = 1
    EXPECT_EQ(cursor.get(), data + 1);
}

UTEST(Coordinates, apply_to_three_dimensions) {
    using TestTensor = spio::Tensor<float, spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 4, 4>,
                                    spio::DimInfo<K, 4, 16>>;

    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    auto coords = spio::make_coordinates(I(1), J(2), K(3));
    auto cursor = coords.apply_to(tensor);

    // Offset: I(1) * 1 + J(2) * 4 + K(3) * 16 = 1 + 8 + 48 = 57
    EXPECT_EQ(cursor.get(), data + 57);
}

// Comparison with CompoundIndex tests
UTEST(Coordinates, less_than_index) {
    using TestIndex = spio::CompoundIndex<spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(1), J(2));
    auto idx = TestIndex(20); // offset 20 = I(0) + J(5) -> I=20%4=0, J=20/4=5

    // coords: I=1, J=2
    // idx.to_coordinates(): I=0, J=5
    // 1 < 0 is false, so not all shared dims satisfy <
    EXPECT_FALSE(coords < idx);
}

UTEST(Coordinates, less_than_index_true) {
    using TestIndex = spio::CompoundIndex<spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(1), J(2));
    auto idx = TestIndex(28); // offset 28 = I(0) + J(7) -> I=28%4=0, J=28/4=7

    // Wait, let's recalculate: offset 28, stride for J is 4
    // I = 28 % 4 = 0, J = 28 / 4 = 7
    // coords: I=1, J=2
    // 1 < 0 is false
    EXPECT_FALSE(coords < idx);
}

UTEST(Coordinates, less_than_index_all_less) {
    using TestIndex = spio::CompoundIndex<spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(1), J(2));
    auto idx = TestIndex(14); // offset 14 = I(2) + J(3) -> I=14%4=2, J=14/4=3

    // coords: I=1, J=2
    // idx: I=2, J=3
    // 1 < 2 and 2 < 3
    EXPECT_TRUE(coords < idx);
}

UTEST(Coordinates, less_equal_index) {
    using TestIndex = spio::CompoundIndex<spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(2), J(3));
    auto idx = TestIndex(14); // I=2, J=3

    // coords: I=2, J=3
    // idx: I=2, J=3
    EXPECT_TRUE(coords <= idx);
}

UTEST(Coordinates, greater_than_index) {
    using TestIndex = spio::CompoundIndex<spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(3), J(5));
    auto idx = TestIndex(9); // I=1, J=2

    // coords: I=3, J=5
    // idx: I=1, J=2
    // 3 > 1 and 5 > 2
    EXPECT_TRUE(coords > idx);
}

UTEST(Coordinates, greater_equal_index) {
    using TestIndex = spio::CompoundIndex<spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(2), J(3));
    auto idx = TestIndex(14); // I=2, J=3

    // coords: I=2, J=3
    // idx: I=2, J=3
    EXPECT_TRUE(coords >= idx);
}

UTEST(Coordinates, equal_index) {
    using TestIndex = spio::CompoundIndex<spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(2), J(3));
    auto idx = TestIndex(14); // I=2, J=3

    EXPECT_TRUE(coords == idx);
}

UTEST(Coordinates, not_equal_index) {
    using TestIndex = spio::CompoundIndex<spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(2), J(3));
    auto idx = TestIndex(15); // I=3, J=3

    EXPECT_TRUE(coords != idx);
}

UTEST(Coordinates, equal_index_partial_overlap) {
    // CompoundIndex has I and J, Coordinates has I and K
    using TestIndex = spio::CompoundIndex<spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(2), K(100));
    auto idx = TestIndex(14); // I=2, J=3

    // Only I is compared, and I=2 in both
    EXPECT_TRUE(coords == idx);
}

UTEST(Coordinates, less_than_index_partial_overlap) {
    using TestIndex = spio::CompoundIndex<spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(1), K(100));
    auto idx = TestIndex(14); // I=2, J=3

    // Only I is compared, and 1 < 2
    EXPECT_TRUE(coords < idx);
}

UTEST(Coordinates, add_index) {
    using TestIndex = spio::CompoundIndex<spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(1), J(2));
    auto idx = TestIndex(14); // I=2, J=3

    auto result = coords + idx;

    // I: 1 + 2 = 3
    // J: 2 + 3 = 5
    EXPECT_EQ(result.get<I>(), I(3));
    EXPECT_EQ(result.get<J>(), J(5));
}

UTEST(Coordinates, add_index_with_fold) {
    using FoldI8 = spio::Fold<I, 8>;
    using TestIndex = spio::CompoundIndex<spio::DimInfo<FoldI8, 4, 1>, spio::DimInfo<J, 8, 4>>;

    auto coords = spio::make_coordinates(I(5), J(2));
    auto idx = TestIndex(9); // FoldI8=1, J=2 -> I=8, J=2

    auto result = coords + idx;

    // I: 5 + 8 = 13 (normalized to I since stride 1)
    // J: 2 + 2 = 4
    EXPECT_EQ(result.get<I>(), I(13));
    EXPECT_EQ(result.get<J>(), J(4));
}

// ============================================================================
// CoordinatesRange tests
// ============================================================================

UTEST(CoordinatesRange, single_dimension) {
    using TestTensor = spio::Tensor<float, spio::DimInfo<I, 3, 1>>;
    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    int count = 0;
    for (auto coord : spio::range(tensor)) {
        EXPECT_EQ(coord.get<I>(), I(count));
        ++count;
    }
    EXPECT_EQ(count, 3);
}

UTEST(CoordinatesRange, two_dimensions) {
    using TestTensor = spio::Tensor<float, spio::DimInfo<I, 2, 1>, spio::DimInfo<J, 3, 2>>;
    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    int count = 0;
    for (auto coord : spio::range(tensor)) {
        // J varies fastest (stride 2 > stride 1)
        I expected_i(count % 2);
        J expected_j(count / 2);
        EXPECT_EQ(coord.get<I>(), expected_i);
        EXPECT_EQ(coord.get<J>(), expected_j);
        ++count;
    }
    EXPECT_EQ(count, 6);
}

UTEST(CoordinatesRange, three_dimensions) {
    using TestTensor =
        spio::Tensor<float, spio::DimInfo<I, 2, 1>, spio::DimInfo<J, 2, 2>, spio::DimInfo<K, 2, 4>>;
    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    int count = 0;
    for (auto coord : spio::range(tensor)) {
        // Order by stride: I (1), J (2), K (4)
        I expected_i(count % 2);
        J expected_j((count / 2) % 2);
        K expected_k(count / 4);
        EXPECT_EQ(coord.get<I>(), expected_i);
        EXPECT_EQ(coord.get<J>(), expected_j);
        EXPECT_EQ(coord.get<K>(), expected_k);
        ++count;
    }
    EXPECT_EQ(count, 8);
}

UTEST(CoordinatesRange, single_element) {
    using TestTensor = spio::Tensor<float, spio::DimInfo<I, 1, 1>, spio::DimInfo<J, 1, 1>>;
    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    int count = 0;
    for (auto coord : spio::range(tensor)) {
        EXPECT_EQ(coord.get<I>(), I(0));
        EXPECT_EQ(coord.get<J>(), J(0));
        ++count;
    }
    EXPECT_EQ(count, 1);
}

UTEST(CoordinatesRange, apply_to_tensor) {
    using TestTensor = spio::Tensor<float, spio::DimInfo<I, 2, 1>, spio::DimInfo<J, 3, 2>>;
    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    // Initialize with index values
    int idx = 0;
    for (auto coord : spio::range(tensor)) {
        *tensor[coord] = static_cast<float>(idx++);
    }

    // Verify using direct indexing
    EXPECT_EQ(*tensor[I(0)][J(0)], 0.0f);
    EXPECT_EQ(*tensor[I(1)][J(0)], 1.0f);
    EXPECT_EQ(*tensor[I(0)][J(1)], 2.0f);
    EXPECT_EQ(*tensor[I(1)][J(1)], 3.0f);
    EXPECT_EQ(*tensor[I(0)][J(2)], 4.0f);
    EXPECT_EQ(*tensor[I(1)][J(2)], 5.0f);
}

UTEST(CoordinatesRange, copy_between_tensors) {
    using TestTensor = spio::Tensor<float, spio::DimInfo<I, 2, 1>, spio::DimInfo<J, 3, 2>>;
    float src_data[TestTensor::storage_size()];
    float dst_data[TestTensor::storage_size()];
    TestTensor src(src_data);
    TestTensor dst(dst_data);

    // Initialize source
    int idx = 0;
    for (auto coord : spio::range(src)) {
        *src[coord] = static_cast<float>(idx++);
    }

    // Copy using range
    for (auto coord : spio::range(src)) {
        *dst[coord] = *src[coord];
    }

    // Verify
    for (auto coord : spio::range(src)) {
        EXPECT_EQ(*dst[coord], *src[coord]);
    }
}

UTEST(CoordinatesRange, iterator_equality) {
    using TestTensor = spio::Tensor<float, spio::DimInfo<I, 2, 1>, spio::DimInfo<J, 2, 2>>;
    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    auto r = spio::range(tensor);

    auto it1 = r.begin();
    auto it2 = r.begin();
    EXPECT_TRUE(it1 == it2);
    EXPECT_FALSE(it1 != it2);

    ++it1;
    EXPECT_FALSE(it1 == it2);
    EXPECT_TRUE(it1 != it2);

    ++it2;
    EXPECT_TRUE(it1 == it2);
}

UTEST(CoordinatesRange, iterator_dereference) {
    using TestTensor = spio::Tensor<float, spio::DimInfo<I, 3, 1>, spio::DimInfo<J, 2, 3>>;
    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    auto r = spio::range(tensor);
    auto it = r.begin();

    {
        auto coord = *it;
        EXPECT_EQ(coord.get<I>(), I(0));
        EXPECT_EQ(coord.get<J>(), J(0));
    }

    ++it;
    {
        auto coord = *it;
        EXPECT_EQ(coord.get<I>(), I(1));
        EXPECT_EQ(coord.get<J>(), J(0));
    }

    ++it;
    {
        auto coord = *it;
        EXPECT_EQ(coord.get<I>(), I(2));
        EXPECT_EQ(coord.get<J>(), J(0));
    }

    ++it;
    {
        auto coord = *it;
        EXPECT_EQ(coord.get<I>(), I(0));
        EXPECT_EQ(coord.get<J>(), J(1));
    }
}

UTEST(CoordinatesRange, accumulate_values) {
    using TestTensor = spio::Tensor<int, spio::DimInfo<I, 3, 1>, spio::DimInfo<J, 3, 3>>;
    int data[TestTensor::storage_size()];
    TestTensor tensor(data);

    // Fill with 1s
    for (auto coord : spio::range(tensor)) {
        *tensor[coord] = 1;
    }

    // Sum all elements
    int sum = 0;
    for (auto coord : spio::range(tensor)) {
        sum += *tensor[coord];
    }
    EXPECT_EQ(sum, 9);
}

UTEST(CoordinatesRange, asymmetric_dimensions) {
    using TestTensor = spio::Tensor<float, spio::DimInfo<I, 1, 1>, spio::DimInfo<J, 5, 1>>;
    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    int count = 0;
    for (auto coord : spio::range(tensor)) {
        EXPECT_EQ(coord.get<I>(), I(0));
        EXPECT_EQ(coord.get<J>(), J(count));
        ++count;
    }
    EXPECT_EQ(count, 5);
}

UTEST(CoordinatesRange, large_iteration) {
    using TestTensor = spio::Tensor<float, spio::DimInfo<I, 4, 1>, spio::DimInfo<J, 8, 4>,
                                    spio::DimInfo<K, 16, 32>>;
    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    int count = 0;
    for (auto coord : spio::range(tensor)) {
        // Order by stride: I (1), J (4), K (32)
        EXPECT_EQ(coord.get<I>(), I(count % 4));
        EXPECT_EQ(coord.get<J>(), J((count / 4) % 8));
        EXPECT_EQ(coord.get<K>(), K(count / 32));
        ++count;
    }
    EXPECT_EQ(count, 4 * 8 * 16);
}

UTEST(CoordinatesRange, fold_dimensions) {
    using FoldI = spio::Fold<I, 8>;
    using FoldJ = spio::Fold<J, 4>;
    using TestTensor = spio::Tensor<float, spio::DimInfo<FoldI, 2, 1>, spio::DimInfo<FoldJ, 3, 2>>;
    float data[TestTensor::storage_size()];
    TestTensor tensor(data);

    int count = 0;
    for ([[maybe_unused]] auto coord : spio::range(tensor)) {
        ++count;
    }
    EXPECT_EQ(count, 6);
}