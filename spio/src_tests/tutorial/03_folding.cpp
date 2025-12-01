#include "tutorial.h"

// Define a Tensor with a folded dimension K and interleaved layout.
// Layout: K8(4) x I(4) x K(8)

/*@spio
[
Tensor("A", dtype.float, Dims(k8=4, i=4, k=8))
]
@spio*/

UTEST(Lesson3, Folding) {

    // Create tensor a.
    A::data_type data[A::storage_size()];
    auto a = A(data);

    // Folded dimension K8 is dimension K folded by stride 8.

    // Dimensions are compatible with their folds:
    EXPECT_EQ(K8(3), K(3 * 8));
    EXPECT_EQ(K8(3) + K(4), K(3 * 8 + 4));

    // Use constant I ..
    auto i = I(2);

    // .. and loop over K in range [0 .. 31] inclusive.
    for (auto k : range(K(32))) {

        // The loop variable has type K.
        static_assert(std::is_same_v<decltype(k), K>, "k should be of type K");

        // Spio accepts logical dimension K
        // and folds it into the tensor's K8 and K dimensions automatically ..
        auto b = a[i][k];

        // .. saving the user from folding it manually.
        auto k8 = K8(k.get() / 8);
        auto km8 = K(k.get() % 8);
        auto c = a[i][k8][km8];

        EXPECT_EQ(b.get(), c.get());
    }
}