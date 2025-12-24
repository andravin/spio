#include "dim_test_common.h"
#include "spio/checkerboard_index.h"
#include "spio/fragment_load_index.h"

using namespace spio;

TEST_DIM(I);
TEST_DIM(K);

using K8 = Fold<K, 8>;
using K64 = Fold<K, 64>;

UTEST(CheckerboardIndex, indices) {
    for (int offset = 0; offset < 256; ++offset) {
        CheckerboardIndex<8, I, K8, OFFSET> idx(offset);
        EXPECT_EQ(idx.get<I>().get(), offset / 2);
        EXPECT_EQ(idx.get<K8>().get(), (offset % 2) ^ ((offset / 8) % 2));
        EXPECT_EQ(idx.get<OFFSET>().get(), offset);
    }
}

UTEST(CheckerboardIndex, offsets) {
    using Checkers = CheckerboardIndex<8, I, K8, OFFSET>;
    for (auto i : range(I(128))) {
        for (auto k8 : range(K8(2))) {
            Checkers idx(i, k8);
            EXPECT_EQ(idx.get<I>(), i);
            EXPECT_EQ(idx.get<K8>(), k8);
            EXPECT_EQ(idx.get<OFFSET>(), Checkers::compute_offset(i, k8));
        }
    }
}

// ============================================================================
// Construction from OffsetDim-compatible dimensions
// ============================================================================

UTEST(CheckerboardFromDim, plain_offset_dim) {
    using Checkers = CheckerboardIndex<8, I, K8, OFFSET>;

    Checkers idx(OFFSET(13));
    EXPECT_EQ(idx.offset(), OFFSET(13));
    EXPECT_EQ(idx.get<I>(), I(13 / 2));                      // 6
    EXPECT_EQ(idx.get<K8>(), K8((13 % 2) ^ ((13 / 8) % 2))); // 1 ^ 1 = 0
}

UTEST(CheckerboardFromDim, fold_of_offset_dim) {
    using Checkers = CheckerboardIndex<8, I, K8, OFFSET>;
    using FoldOffset8 = Fold<OFFSET, 8>;

    // Fold<OFFSET, 8>(2) -> base value 16 -> OFFSET(16)
    Checkers idx(FoldOffset8(2));
    EXPECT_EQ(idx.offset(), OFFSET(16));
    EXPECT_EQ(idx.get<I>(), I(8));
}

// ============================================================================
// Derived dimension interface
// ============================================================================

UTEST(CheckerboardDerivedDims, type_aliases) {
    using Checkers = CheckerboardIndex<32, I, K8, OFFSET>;

    // Verify size constants
    static_assert(Checkers::size == 32);
    static_assert(Checkers::num_colors == 2);
    static_assert(Checkers::num_pairs == 16);

    // Verify input_dims contains DimSize for PairDim and ColorDim
    using InputDims = Checkers::input_dims;
    static_assert(detail::tuple_size<InputDims>::value == 2);

    using PairDimSize = detail::tuple_element_t<0, InputDims>;
    using ColorDimSize = detail::tuple_element_t<1, InputDims>;

    static_assert(detail::is_same<typename PairDimSize::dim_type, I>::value);
    static_assert(PairDimSize::size == 16);
    static_assert(detail::is_same<typename ColorDimSize::dim_type, K8>::value);
    static_assert(ColorDimSize::size == 2);

    // Verify output_dims contains DimSize for OffsetDim
    using OutputDims = Checkers::output_dims;
    static_assert(detail::tuple_size<OutputDims>::value == 1);

    using OffsetDimSize = detail::tuple_element_t<0, OutputDims>;
    static_assert(detail::is_same<typename OffsetDimSize::dim_type, OFFSET>::value);
    static_assert(OffsetDimSize::size == 32);

    EXPECT_TRUE(true); // Static asserts passed
}

UTEST(CheckerboardDerivedDims, is_derived_dim_trait) {
    using Checkers = CheckerboardIndex<32, I, K8, OFFSET>;

    // CheckerboardIndex should be detected as a derived dimension type
    static_assert(detail::is_derived_dim_v<Checkers>);

    // DimInfo should NOT be detected as a derived dimension type
    static_assert(!detail::is_derived_dim_v<DimInfo<I, 16, 1>>);

    // Plain types should NOT be detected as derived dimensions
    static_assert(!detail::is_derived_dim_v<int>);
    static_assert(!detail::is_derived_dim_v<I>);

    EXPECT_TRUE(true); // Static asserts passed
}

UTEST(CheckerboardDerivedDims, compute_coordinates) {
    using Checkers = CheckerboardIndex<32, I, K8, OFFSET>;

    // Test compute_coordinates returns correct output coordinates
    for (auto i : range(I(16))) {
        for (auto k8 : range(K8(2))) {
            auto input_coords = make_coordinates(i, k8);
            auto output_coords = Checkers::compute_coordinates(input_coords);
            auto expected_offset = Checkers::compute_offset(i, k8);
            EXPECT_EQ(output_coords.template get<OFFSET>(), expected_offset);
        }
    }
}

UTEST(CheckerboardDerivedDims, matching_logic) {
    using Checkers = CheckerboardIndex<32, I, K8, OFFSET>;

    // Coordinates with both I and K8 should match
    using CoordsIK = Coordinates<I, K8>;
    static_assert(detail::derived_dim_matches_coords<Checkers, CoordsIK::dims_tuple>::value);

    // Coordinates with I and K (base of K8) should also match
    using CoordsIKbase = Coordinates<I, K>;
    static_assert(detail::derived_dim_matches_coords<Checkers, CoordsIKbase::dims_tuple>::value);

    // Coordinates with I and K64 should NOT match:
    // DimSize<K8, 2> needs K8(0)..K8(1). K64(1).fold<8>() = K8(8), and 8 > 1.
    using CoordsIK64 = Coordinates<I, K64>;
    static_assert(!detail::derived_dim_matches_coords<Checkers, CoordsIK64::dims_tuple>::value);

    // Coordinates with only I should NOT match (missing K8/K)
    using CoordsI = Coordinates<I>;
    static_assert(!detail::derived_dim_matches_coords<Checkers, CoordsI::dims_tuple>::value);

    // Coordinates with only K8 should NOT match (missing I)
    using CoordsK = Coordinates<K8>;
    static_assert(!detail::derived_dim_matches_coords<Checkers, CoordsK::dims_tuple>::value);

    // Coordinates with unrelated dims should NOT match
    using CoordsOffset = Coordinates<OFFSET>;
    static_assert(!detail::derived_dim_matches_coords<Checkers, CoordsOffset::dims_tuple>::value);

    // Test find_matching_derived_dim
    using DerivedDims = detail::tuple<Checkers>;

    using FoundWithIK = detail::find_matching_derived_dim_t<DerivedDims, CoordsIK::dims_tuple>;
    static_assert(detail::is_same<FoundWithIK, Checkers>::value);

    using FoundWithI = detail::find_matching_derived_dim_t<DerivedDims, CoordsI::dims_tuple>;
    static_assert(detail::is_same<FoundWithI, void>::value);

    EXPECT_TRUE(true); // Static asserts passed
}

UTEST(CheckerboardDerivedDims, apply_derived_dim) {
    using Checkers = CheckerboardIndex<32, I, K8, OFFSET>;

    // Test apply_derived_dim with exact input types
    for (auto i : range(I(16))) {
        for (auto k8 : range(K8(2))) {
            auto input_coords = make_coordinates(i, k8);
            auto output_coords = detail::apply_derived_dim<Checkers>(input_coords);
            auto expected_offset = Checkers::compute_offset(i, k8);
            EXPECT_EQ(output_coords.template get<OFFSET>(), expected_offset);
        }
    }

    // Test apply_derived_dim with base K dimension (should fold to K8)
    // K folds to K8 with stride 8: K(0)..K(7) -> K8(0), K(8)..K(15) -> K8(1)
    for (auto i : range(I(16))) {
        for (int color = 0; color < 2; ++color) {
            auto k = K(color * 8); // K(0) or K(8)
            auto input_coords = make_coordinates(i, k);
            auto output_coords = detail::apply_derived_dim<Checkers>(input_coords);
            auto expected_offset = Checkers::compute_offset(i, K8(color));
            EXPECT_EQ(output_coords.template get<OFFSET>(), expected_offset);
        }
    }
    // Now test wrap-around (modulo) behavior for K8 input
    // K8 has size 2, so valid values are K8(0) and K8(1)
    // Test projecting K coordinates that exceed this range
    for (auto i : range(I(16))) {
        for (int k_val = 0; k_val < 16; ++k_val) {
            auto k = K(k_val); // K(0) to K(15)
            auto input_coords = make_coordinates(i, k);
            int folded = (k_val / 8); // 0 or 1
            int wrapped = folded % 2; // modulo K8 size
            auto expected_offset = Checkers::compute_offset(i, K8(wrapped));
            auto output_coords = detail::apply_derived_dim<Checkers>(input_coords);
            EXPECT_EQ(output_coords.template get<OFFSET>(), expected_offset);
        }
    }
}
