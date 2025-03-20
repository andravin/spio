#include "utest.h"
#include "spio/checkerboard_index.h"

using namespace spio;

class I : public Dim<I>
{
    using Dim::Dim;
};
class K : public Dim<K> { using Dim::Dim; };
class OFFSET : public Dim<OFFSET> { using Dim::Dim; };
using K8 = Fold<K, 8>;

UTEST(CheckerboardIndex, indices)
{
    for (int offset = 0; offset < 256; ++offset)
    {
        CheckerboardIndex<8, I, K8, OFFSET> idx(offset);
        EXPECT_EQ(idx.get<I>().get(), offset / 2);
        EXPECT_EQ(idx.get<K8>().get(), (offset % 2) ^ ((offset / 8) % 2));
        EXPECT_EQ(idx.get<OFFSET>().get(), offset);
    }
}

UTEST(CheckerboardIndex, offsets)
{
    using Checkers = CheckerboardIndex<8, I, K8, OFFSET>;
    for (auto i : range(I(128)))
    {
        for (auto k8 : range(K8(2)))
        {
            Checkers idx(i, k8);
            EXPECT_EQ(idx.get<I>().get(), i.get());
            EXPECT_EQ(idx.get<K8>().get(), k8.get());
            EXPECT_EQ(idx.get<OFFSET>().get(), Checkers::compute_offset(i.get(), k8.get()));
        }
    }
}