#include "dim_test_common.h"
#include "spio/fragment_load_index.h"
#include "spio/fragment_index.h"
#include "spio/compound_index.h"
#include "spio/checkerboard_index.h"

using namespace spio;

class I : public Dim<I> {
    using Dim::Dim;
};

class J : public Dim<J> {
    using Dim::Dim;
};

class K : public Dim<K> {
    using Dim::Dim;
};

using K8 = Fold<K, 8>;
using K2 = Fold<K, 2>;
using K2M4 = Module<K, 4, 2>;

using J2 = Fold<J, 2>;
using J8 = Fold<J, 8>;
using J2M4 = Module<J, 4, 2>;

UTEST(MMA_A_M16_K8_F16_LoadIndex, indices) {
    for (int lane = 0; lane < 32; ++lane) {
        MMA_A_M16_K8_F16_LoadIndex<I, K> idx(lane);
        EXPECT_EQ(idx.get<I>().get(), lane % 16);
        EXPECT_EQ(idx.get<K8>().get(), 0);
    }
}

UTEST(MMA_A_M16_K16_F16_LoadIndex, indices) {
    for (int lane = 0; lane < 32; ++lane) {
        MMA_A_M16_K16_F16_LoadIndex<I, K> idx(lane);
        EXPECT_EQ(idx.get<I>().get(), lane % 16);
        EXPECT_EQ(idx.get<K8>().get(), lane / 16);
    }
}

UTEST(MMA_B_N8_K8_F16_LoadIndex, indices) {
    for (int lane = 0; lane < 32; ++lane) {
        MMA_B_N8_K8_F16_LoadIndex<K, J> idx(lane);
        EXPECT_EQ(idx.get<J>().get(), lane % 8);
        EXPECT_EQ(idx.get<K8>().get(), 0);
    }
}

UTEST(MMA_B_N8_K16_F16_LoadIndex, indices) {
    for (int lane = 0; lane < 32; ++lane) {
        MMA_B_N8_K16_F16_LoadIndex<K, J> idx(lane);
        EXPECT_EQ(idx.get<J>().get(), lane % 8);
        EXPECT_EQ(idx.get<K8>().get(), (lane / 8) % 2);
    }
}

UTEST(MMA_B_N16_K16_F16_LoadIndex, indices) {
    for (int lane = 0; lane < 32; ++lane) {
        MMA_B_N16_K16_F16_LoadIndex<K, J> idx(lane);
        EXPECT_EQ(idx.get<J>().get(), (lane % 8) + lane / 16 * 8);
        EXPECT_EQ(idx.get<K8>().get(), (lane / 8) % 2);
    }
}

UTEST(MMA_A_88_F16_Index, indices) {
    for (int lane = 0; lane < 32; ++lane) {
        for (int f = 0; f < 4; ++f) {
            MMA_A_88_F16_Index<I, K> idx(lane);
            EXPECT_EQ(idx.get<I>(f).get(), lane / 4 + (f % 2) * 8);
            EXPECT_EQ(idx.get<K2>(f).get(), (lane % 4) + f / 2 * 4);
            EXPECT_EQ(idx.get<K8>(f).get(), f / 2);
            EXPECT_EQ(idx.get<K2M4>().get(), lane % 4);
        }
    }
}

UTEST(MMA_B_88_F16_Index, indices) {
    for (int lane = 0; lane < 32; ++lane) {
        for (int f = 0; f < 4; ++f) {
            MMA_B_88_F16_Index<K, J> idx(lane);
            EXPECT_EQ(idx.get<J>(f).get(), (lane / 4) + (f / 2) * 8);
            EXPECT_EQ(idx.get<K2>(f).get(), (lane % 4) + (f % 2) * 4);
            EXPECT_EQ(idx.get<K8>(f).get(), f % 2);
            EXPECT_EQ(idx.get<K2M4>().get(), lane % 4);
        }
    }
}

UTEST(MMA_C_M16_N16_F32_Index, indices) {
    for (int lane = 0; lane < 32; ++lane) {
        for (int f = 0; f < 4; ++f) {
            MMA_C_88_F32_Index<I, J> idx(lane);
            EXPECT_EQ(idx.get<I>(f).get(), (lane / 4) + (f % 2) * 8);
            EXPECT_EQ(idx.get<J2>(f).get(), (lane % 4) + f / 2 * 4);
            EXPECT_EQ(idx.get<J8>(f).get(), f / 2);
            EXPECT_EQ(idx.get<J2M4>().get(), lane % 4);
        }
    }
}

// ============================================================================
// Construction from CompoundIndex with LANE dimension
// ============================================================================

UTEST(MMA_A_M16_K16_F16_LoadIndex, from_compound_index_with_lane) {
    // Simulates the kernel pattern:
    // auto compute_idx = ComputeIndex(threadIdx.x);
    // auto a_load_idx = A_Tile::data_type::load_index_type(compute_idx);

    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);

        // Construct load_index_type from CompoundIndex via coordinates()
        MMA_A_M16_K16_F16_LoadIndex<I, K> load_idx(compute_idx);

        // Should extract LANE value and compute I and K8
        EXPECT_EQ(load_idx.get<I>().get(), lane % 16);
        EXPECT_EQ(load_idx.get<K8>().get(), lane / 16);
    }
}

UTEST(MMA_A_M16_K16_F16_LoadIndex, from_compound_index_with_extra_dims) {
    // ComputeIndex typically has WARP dimensions too
    using ComputeIdx = CompoundIndex<DimInfo<I, 4, 32>, DimInfo<J, 4, 128>, DimInfo<LANE, 32, 1>>;

    for (int warp_i = 0; warp_i < 4; ++warp_i) {
        for (int warp_j = 0; warp_j < 4; ++warp_j) {
            for (int lane = 0; lane < 32; ++lane) {
                int offset = warp_i * 32 + warp_j * 128 + lane;
                ComputeIdx compute_idx(offset);

                // load_index_type should only use the LANE component
                MMA_A_M16_K16_F16_LoadIndex<I, K> load_idx(compute_idx);

                EXPECT_EQ(load_idx.get<I>().get(), lane % 16);
                EXPECT_EQ(load_idx.get<K8>().get(), lane / 16);
            }
        }
    }
}

UTEST(MMA_B_N16_K16_F16_LoadIndex, from_compound_index_with_lane) {
    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);

        MMA_B_N16_K16_F16_LoadIndex<K, J> load_idx(compute_idx);

        EXPECT_EQ(load_idx.get<J>().get(), (lane % 8) + lane / 16 * 8);
        EXPECT_EQ(load_idx.get<K8>().get(), (lane / 8) % 2);
    }
}

UTEST(MMA_A_M16_K8_F16_LoadIndex, from_compound_index_with_lane) {
    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);

        MMA_A_M16_K8_F16_LoadIndex<I, K> load_idx(compute_idx);

        EXPECT_EQ(load_idx.get<I>().get(), lane % 16);
        EXPECT_EQ(load_idx.get<K8>().get(), 0);
    }
}

UTEST(MMA_B_N8_K16_F16_LoadIndex, from_compound_index_with_lane) {
    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);

        MMA_B_N8_K16_F16_LoadIndex<K, J> load_idx(compute_idx);

        EXPECT_EQ(load_idx.get<J>().get(), lane % 8);
        EXPECT_EQ(load_idx.get<K8>().get(), (lane / 8) % 2);
    }
}

// ============================================================================
// Construction from CompoundIndex with Module<LANE, ...>
// ============================================================================

UTEST(MMA_A_M16_K16_F16_LoadIndex, from_compound_index_with_module_lane) {
    // When CompoundIndex::coordinates() is called, it returns Module<LANE, size, stride>
    // The load_index_type should handle this correctly via dimensional projection
    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);

        // Get coordinates which contains Module<LANE, 32, 1>
        auto coords = compute_idx.coordinates();

        // Construct load_index_type from coordinates directly
        MMA_A_M16_K16_F16_LoadIndex<I, K> load_idx(coords);

        EXPECT_EQ(load_idx.get<I>().get(), lane % 16);
        EXPECT_EQ(load_idx.get<K8>().get(), lane / 16);
    }
}

UTEST(MMA_A_M16_K16_F16_LoadIndex, from_compound_index_with_fold_lane) {
    // Test with Fold<LANE, N> in the CompoundIndex
    using FoldLane8 = Fold<LANE, 8>;
    using ComputeIdx = CompoundIndex<DimInfo<FoldLane8, 4, 1>>;

    for (int fold_val = 0; fold_val < 4; ++fold_val) {
        ComputeIdx compute_idx(fold_val);

        // Fold<LANE, 8>(fold_val) -> base value = fold_val * 8
        int lane = fold_val * 8;

        MMA_A_M16_K16_F16_LoadIndex<I, K> load_idx(compute_idx);

        EXPECT_EQ(load_idx.get<I>().get(), lane % 16);
        EXPECT_EQ(load_idx.get<K8>().get(), lane / 16);
    }
}

// ============================================================================
// Roundtrip tests: CompoundIndex -> load_index_type -> CheckerboardIndex
// ============================================================================

UTEST(LoadIndexToCheckerboard, mma_a_full_chain) {
    // This mirrors the kernel pattern:
    // auto compute_idx = ComputeIndex(threadIdx.x);
    // auto a_load_idx = A_Tile::data_type::load_index_type(compute_idx);
    // auto smem_a_checkers = SmemA_Checkers(a_load_idx);

    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;
    using Checkers = CheckerboardIndex<8, I, K8, OFFSET>;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);
        MMA_A_M16_K16_F16_LoadIndex<I, K> load_idx(compute_idx);
        Checkers checkers(load_idx);

        // Verify the full chain produces correct values
        I expected_i = load_idx.get<I>();
        K8 expected_k8 = load_idx.get<K8>();

        EXPECT_EQ(checkers.get<I>(), expected_i);
        EXPECT_EQ(checkers.get<K8>(), expected_k8);
        EXPECT_EQ(checkers.offset(), Checkers::compute_offset(expected_i, expected_k8));
    }
}

UTEST(LoadIndexToCheckerboard, mma_b_full_chain) {
    using ComputeIdx = CompoundIndex<DimInfo<LANE, 32, 1>>;
    using Checkers = CheckerboardIndex<8, J, K8, OFFSET>;

    for (int lane = 0; lane < 32; ++lane) {
        ComputeIdx compute_idx(lane);
        MMA_B_N16_K16_F16_LoadIndex<K, J> load_idx(compute_idx);
        Checkers checkers(load_idx);

        J expected_j = load_idx.get<J>();
        K8 expected_k8 = load_idx.get<K8>();

        EXPECT_EQ(checkers.get<J>(), expected_j);
        EXPECT_EQ(checkers.get<K8>(), expected_k8);
        EXPECT_EQ(checkers.offset(), Checkers::compute_offset(expected_j, expected_k8));
    }
}