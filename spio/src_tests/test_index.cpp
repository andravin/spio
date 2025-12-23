#include <vector>
#include <algorithm>

#include "dim_test_types.h"
#include "spio/compound_index.h"
#include "spio/coordinates.h"

using namespace spio;

UTEST(Index1D, get) {
    using Idx = CompoundIndex<DimInfo<I, 16, 1>>;
    for (unsigned offset = 0; offset < 16; ++offset) {
        Idx idx(offset);
        EXPECT_EQ(idx.get<I>().get(), offset);
    }
}

UTEST(Index2D, get) {
    // I x J matrix with size 32 x 8.
    using Idx = CompoundIndex<DimInfo<I, 32, 8>, DimInfo<J, 8, 1>>;
    for (unsigned offset = 0; offset < 256; ++offset) {
        Idx idx(offset);
        EXPECT_EQ(idx.get<I>().get(), offset / 8);
        EXPECT_EQ(idx.get<J>().get(), offset % 8);
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
        EXPECT_EQ(idx.get<I>().get(), offset / (8 * 4));
        EXPECT_EQ(idx.get<J>().get(), (offset / 4) % 8);
        EXPECT_EQ(idx.get<K>().get(), offset % 4);
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

    // Index with two folds of the same base dimension (hierarchical)
    // I8 (coarse, stride 8) and I (fine, stride 1) are complementary folds
    using I8 = Fold<I, 8>;
    using IdxFolded = CompoundIndex<DimInfo<I8, 8, 8>, DimInfo<I, 8, 1>>;
    // Total size is product: 8 * 8 = 64
    // This represents 8 groups of 8 elements
    EXPECT_EQ(IdxFolded::total_size, 8 * 8);
    EXPECT_EQ(IdxFolded::size(), 8 * 8);
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
    EXPECT_EQ(idx_j.offset(), OFFSET(5));
}

UTEST(IndexCast, preserves_offset) {
    // Casting should preserve the offset value
    using IdxI = CompoundIndex<DimInfo<I, 32, 8>, DimInfo<J, 8, 1>>;

    for (unsigned offset = 0; offset < 256; ++offset) {
        IdxI idx(offset);
        auto casted = idx.cast<I, K>(); // Replace I with K
        EXPECT_EQ(casted.offset(), OFFSET(offset));
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
    EXPECT_EQ(casted.offset(), OFFSET(3));
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
    EXPECT_EQ(casted.offset(), OFFSET(5));
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
    EXPECT_EQ(casted.offset(), OFFSET(6));
    EXPECT_EQ(casted.get<J>().get(), idx.get<I>().get());
    EXPECT_EQ(casted.get<FoldJ8>().get(), idx.get<FoldI8>().get());
}

UTEST(IndexCast, unrelated_dimensions_unchanged) {
    // Dimensions with different base should not be affected
    using IdxIJ = CompoundIndex<DimInfo<I, 8, 4>, DimInfo<J, 4, 1>>;

    IdxIJ idx(15);
    auto casted = idx.cast<I, K>(); // Only replace I with K, J unchanged

    EXPECT_EQ(casted.offset(), OFFSET(15));
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
    EXPECT_EQ(casted.offset(), OFFSET(10));
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
    EXPECT_EQ(casted.offset(), OFFSET(1));
    EXPECT_EQ(casted.get<NestedFoldJ>().get(), idx.get<NestedFoldI>().get());
}

UTEST(IndexCast, chained_casts) {
    // Multiple casts in sequence
    using IdxI = CompoundIndex<DimInfo<I, 16, 1>>;

    IdxI idx(7);
    auto idx_j = idx.cast<I, J>();
    auto idx_k = idx_j.cast<J, K>();

    EXPECT_EQ(idx_k.get<K>(), K(7));
    EXPECT_EQ(idx_k.offset(), OFFSET(7));
}

// ============================================================================
// Construction from OFFSET-compatible dimensions
// ============================================================================

UTEST(IndexFromDim, plain_offset) {
    using Idx = CompoundIndex<DimInfo<I, 16, 1>>;

    Idx idx(OFFSET(5));
    EXPECT_EQ(idx.offset(), OFFSET(5));
    EXPECT_EQ(idx.get<I>(), I(5));
}

UTEST(IndexFromDim, fold_of_offset) {
    using Idx = CompoundIndex<DimInfo<I, 16, 1>>;
    using FoldOffset8 = Fold<OFFSET, 8>;

    // Fold<OFFSET, 8>(3) -> base value 3 * 8 = 24 -> OFFSET(24)
    Idx idx(FoldOffset8(3));
    EXPECT_EQ(idx.offset(), OFFSET(24));
    EXPECT_EQ(idx.get<I>(), I(24 % 16)); // 24 % 16 = 8
}

UTEST(IndexFromDim, module_of_offset) {
    using Idx = CompoundIndex<DimInfo<I, 16, 1>>;
    using ModuleOffset = Module<OFFSET, 32, 1>;

    ModuleOffset mod(5);
    Idx idx(mod);
    EXPECT_EQ(idx.offset(), OFFSET(5));
    EXPECT_EQ(idx.get<I>(), I(5));
}

// ============================================================================
// Construction from Coordinates containing OFFSET
// ============================================================================

UTEST(IndexFromCoords, single_offset_in_coords) {
    using Idx = CompoundIndex<DimInfo<I, 16, 1>>;

    auto coords = make_coordinates(OFFSET(7));
    Idx idx(coords);

    EXPECT_EQ(idx.offset(), OFFSET(7));
    EXPECT_EQ(idx.get<I>(), I(7));
}

UTEST(IndexFromCoords, fold_of_offset_in_coords) {
    using Idx = CompoundIndex<DimInfo<I, 16, 1>>;
    using FoldOffset8 = Fold<OFFSET, 8>;

    // Fold<OFFSET, 8>(2) -> base value 2 * 8 = 16 -> OFFSET(16)
    auto coords = make_coordinates(FoldOffset8(2));
    Idx idx(coords);

    EXPECT_EQ(idx.offset(), OFFSET(16));
    EXPECT_EQ(idx.get<I>(), I(0)); // 16 % 16 = 0
}

UTEST(IndexFromCoords, multiple_offset_folds_summed) {
    using Idx = CompoundIndex<DimInfo<I, 32, 8>, DimInfo<J, 8, 1>>;
    using FoldOffset8 = Fold<OFFSET, 8>;

    // OFFSET(5) + Fold<OFFSET, 8>(2) -> base values 5 + 16 = 21
    auto coords = make_coordinates(OFFSET(5), FoldOffset8(2));
    Idx idx(coords);

    EXPECT_EQ(idx.offset(), OFFSET(21));
    EXPECT_EQ(idx.get<I>(), I(21 / 8)); // 2
    EXPECT_EQ(idx.get<J>(), J(21 % 8)); // 5
}

UTEST(IndexFromCoords, offset_with_unrelated_dims_ignored) {
    using Idx = CompoundIndex<DimInfo<I, 16, 1>>;

    // Only OFFSET should contribute, I and J are unrelated to OFFSET base
    auto coords = make_coordinates(OFFSET(9), I(100), J(200));
    Idx idx(coords);

    EXPECT_EQ(idx.offset(), 9);
    EXPECT_EQ(idx.get<I>(), I(9));
}

UTEST(IndexFromCoords, different_fold_strides_normalized) {
    using Idx = CompoundIndex<DimInfo<I, 64, 1>>;
    using FoldOffset4 = Fold<OFFSET, 4>;
    using FoldOffset16 = Fold<OFFSET, 16>;

    // Fold<OFFSET, 4>(3) -> base 12
    // Fold<OFFSET, 16>(1) -> base 16
    // Total: 12 + 16 = 28
    auto coords = make_coordinates(FoldOffset4(3), FoldOffset16(1));
    Idx idx(coords);

    EXPECT_EQ(idx.offset(), OFFSET(28));
    EXPECT_EQ(idx.get<I>(), I(28));
}

// ============================================================================
// Construction from types with coordinates() method
// ============================================================================

UTEST(IndexFromHasCoords, from_another_compound_index_with_offset) {
    // A hypothetical index that stores OFFSET in its coordinates
    using SourceIdx = CompoundIndex<DimInfo<OFFSET, 32, 1>>;
    using TargetIdx = CompoundIndex<DimInfo<I, 16, 2>, DimInfo<J, 2, 1>>;

    SourceIdx source(25);
    // source.coordinates() returns Coordinates<Module<OFFSET, 32, 1>>
    // which contains OFFSET, so it should work
    TargetIdx target(source);

    EXPECT_EQ(target.offset(), OFFSET(25));
    EXPECT_EQ(target.get<I>(), I(25 / 2)); // 12
    EXPECT_EQ(target.get<J>(), J(25 % 2)); // 1
}

// ============================================================================
// Edge cases and roundtrip
// ============================================================================

UTEST(IndexFromCoords, zero_offset) {
    using Idx = CompoundIndex<DimInfo<I, 16, 1>>;

    auto coords = make_coordinates(OFFSET(0));
    Idx idx(coords);

    EXPECT_EQ(idx.offset(), OFFSET(0));
    EXPECT_EQ(idx.get<I>(), I(0));
}

UTEST(IndexFromCoords, max_offset_for_index) {
    using Idx = CompoundIndex<DimInfo<I, 8, 4>, DimInfo<J, 4, 1>>;
    constexpr OFFSET max_offset = 8 * 4 - 1; // 31

    auto coords = make_coordinates(max_offset);
    Idx idx(coords);

    EXPECT_EQ(idx.offset(), max_offset);
    EXPECT_EQ(idx.get<I>(), I(7));
    EXPECT_EQ(idx.get<J>(), J(3));
}

UTEST(IndexFromDim, roundtrip_offset_to_coords_to_offset) {
    using Idx = CompoundIndex<DimInfo<OFFSET, 64, 1>>;

    for (int i = 0; i < 64; i += 7) {
        Idx original(i);
        auto coords = original.coordinates();
        // coordinates() returns Module<OFFSET, 64, 1> which is OFFSET-compatible
        Idx reconstructed(coords);

        EXPECT_EQ(reconstructed.offset(), OFFSET(i));
    }
}

// ============================================================================
// Partition tests - cooperative iteration over index space
// ============================================================================

UTEST(IndexPartition, basic_partition_by_lane) {
    // Partition a 64-element index space across 32 lanes
    using TargetIdx = CompoundIndex<DimInfo<I, 64, 1>>;
    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;

    // Each lane should handle 2 elements: lane 0 -> {0, 32}, lane 1 -> {1, 33}, etc.
    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);

        std::vector<int> offsets;
        for (auto idx : TargetIdx::partition<LANE>(compute_idx)) {
            offsets.push_back(idx.offset().get());
        }

        ASSERT_EQ(offsets.size(), 2);
        EXPECT_EQ(offsets[0], lane);
        EXPECT_EQ(offsets[1], lane + 32);
    }
}

UTEST(IndexPartition, partition_with_remainder) {
    // Partition a 40-element index space across 32 lanes
    // Lanes 0-7 get 2 elements, lanes 8-31 get 1 element
    using TargetIdx = CompoundIndex<DimInfo<I, 40, 1>>;
    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);

        std::vector<int> offsets;
        for (auto idx : TargetIdx::partition<LANE>(compute_idx)) {
            offsets.push_back(idx.offset().get());
        }

        if (lane < 8) {
            // Lanes 0-7 handle offsets lane and lane+32
            ASSERT_EQ(offsets.size(), 2);
            EXPECT_EQ(offsets[0], lane);
            EXPECT_EQ(offsets[1], lane + 32);
        } else {
            // Lanes 8-31 only handle offset lane
            ASSERT_EQ(offsets.size(), 1);
            EXPECT_EQ(offsets[0], lane);
        }
    }
}

UTEST(IndexPartition, partition_smaller_than_stride) {
    // Partition a 16-element index space across 32 lanes
    // Only lanes 0-15 get any work
    using TargetIdx = CompoundIndex<DimInfo<I, 16, 1>>;
    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);

        std::vector<int> offsets;
        for (auto idx : TargetIdx::partition<LANE>(compute_idx)) {
            offsets.push_back(idx.offset().get());
        }

        if (lane < 16) {
            ASSERT_EQ(offsets.size(), 1);
            EXPECT_EQ(offsets[0], lane);
        } else {
            ASSERT_EQ(offsets.size(), 0);
        }
    }
}

UTEST(IndexPartition, partition_2d_index) {
    // Partition a 2D index space (8x8 = 64 elements) across 32 lanes
    using TargetIdx = CompoundIndex<DimInfo<I, 8, 8>, DimInfo<J, 8, 1>>;
    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);

        std::vector<int> offsets;
        for (auto idx : TargetIdx::partition<LANE>(compute_idx)) {
            offsets.push_back(idx.offset().get());
            // Verify get<I>() and get<J>() work correctly
            EXPECT_EQ(idx.get<I>().get(), idx.offset().get() / 8);
            EXPECT_EQ(idx.get<J>().get(), idx.offset().get() % 8);
        }

        ASSERT_EQ(offsets.size(), 2);
        EXPECT_EQ(offsets[0], lane);
        EXPECT_EQ(offsets[1], lane + 32);
    }
}

UTEST(IndexPartition, partition_with_extra_dims_in_compute_idx) {
    // ComputeIndex has extra dimensions (like warp indices) that should be ignored
    using TargetIdx = CompoundIndex<DimInfo<I, 64, 1>>;
    using ComputeIdx = CompoundIndex<DimInfo<I, 4, 32>, DimInfo<J, 4, 128>, DimInfo<LANE, 32, 1>>;

    // Test a few representative compute indices with different warp coordinates
    // The partition should only use LANE, not the warp indices
    for (int warp_i = 0; warp_i < 4; ++warp_i) {
        for (int lane = 0; lane < 32; ++lane) {
            int compute_offset = warp_i * 32 + lane; // Assuming warp_j = 0
            ComputeIdx compute_idx(compute_offset);

            std::vector<int> offsets;
            for (auto idx : TargetIdx::partition<LANE>(compute_idx)) {
                offsets.push_back(idx.offset().get());
            }

            // Should always get same result for same lane, regardless of warp_i
            ASSERT_EQ(offsets.size(), 2);
            EXPECT_EQ(offsets[0], lane);
            EXPECT_EQ(offsets[1], lane + 32);
        }
    }
}

UTEST(IndexPartition, partition_exact_multiple) {
    // Partition exactly 32 elements across 32 lanes - each lane gets exactly 1
    using TargetIdx = CompoundIndex<DimInfo<I, 32, 1>>;
    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);

        std::vector<int> offsets;
        for (auto idx : TargetIdx::partition<LANE>(compute_idx)) {
            offsets.push_back(idx.offset().get());
        }

        ASSERT_EQ(offsets.size(), 1);
        EXPECT_EQ(offsets[0], lane);
    }
}

UTEST(IndexPartition, partition_large_index) {
    // Partition a large index space (1024 elements) across 32 lanes
    using TargetIdx = CompoundIndex<DimInfo<I, 1024, 1>>;
    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);

        int count = 0;
        int expected_offset = lane;
        for (auto idx : TargetIdx::partition<LANE>(compute_idx)) {
            EXPECT_EQ(idx.offset().get(), expected_offset);
            expected_offset += 32;
            ++count;
        }

        EXPECT_EQ(count, 32); // 1024 / 32 = 32 elements per lane
    }
}

UTEST(IndexPartition, all_offsets_covered) {
    // Verify that all offsets are covered exactly once across all lanes
    using TargetIdx = CompoundIndex<DimInfo<I, 100, 1>>;
    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;

    std::vector<int> all_offsets;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);
        for (auto idx : TargetIdx::partition<LANE>(compute_idx)) {
            all_offsets.push_back(idx.offset().get());
        }
    }

    // Sort and verify all offsets 0-99 are present exactly once
    std::sort(all_offsets.begin(), all_offsets.end());
    ASSERT_EQ(all_offsets.size(), 100);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(all_offsets[i], i);
    }
}
