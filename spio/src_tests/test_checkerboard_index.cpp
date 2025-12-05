#include "dim_test_common.h"
#include "spio/checkerboard_index.h"
#include "spio/fragment_load_index.h"

using namespace spio;

TEST_DIM(I);
TEST_DIM(K);

using K8 = Fold<K, 8>;

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
