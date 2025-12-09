#include "tutorial.h"

// Define a Tensor with a folded dimension K and interleaved layout.
// Layout: K8(4) x I(4) x K(8)

/*@spio
[
Tensor("A", dtype.float, Dims(k8=4, i=4, k=-1))
]
@spio*/

UTEST(Lesson3, Folding) {

    // Create tensor a.
    A::data_type data[A::storage_size()];
    auto a = A(data);

    // Folded dimension K8 is dimension K folded by stride 8.

    // Dimensions are compatible with their folds:
    EXPECT_TRUE(K8(3) == K(3 * 8));
    EXPECT_TRUE(K8(3) + K(4) == K(3 * 8 + 4));

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

        EXPECT_TRUE(b.get() == c.get());
    }
}

UTEST(Lesson3, CrossFoldCarry) {

    // Create tensor a.
    A::data_type data[A::storage_size()];
    for (int j = 0; j < A::storage_size(); ++j) {
        data[j] = static_cast<float>(j);
    }
    auto a = A(data);

    // Use constant I.
    auto i = I(1);

    // Spio accumulates subscripts in logical coordinates before folding.
    // This means repeated subscripts are equivalent to their sum.

    // Example: K(4) + K(4) = K(4 + 4), which carries into K8.
    EXPECT_TRUE(*a[i][K(4)][K(4)] == *a[i][K(4 + 4)]);

    // More generally, a[e][f][g] == a[e + f + g] for any dimension values.
    for (int e = 0; e < 8; ++e) {
        for (int f = 0; f < 8; ++f) {
            // Subscripting separately should equal subscripting with the sum.
            EXPECT_TRUE(*a[i][K(e)][K(f)] == *a[i][K(e + f)]);
        }
    }

    // This also works when the sum crosses fold boundaries.
    // K(7) + K(5) = K(12) = K8(1) + K(4)
    EXPECT_TRUE(*a[i][K(7)][K(5)] == *a[i][K(7 + 5)]);
    EXPECT_TRUE(*a[i][K(7)][K(5)] == *a[i][K8(1)][K(4)]);

    // And with cursor stepping:
    auto cursor = a[i];
    cursor.step(K(7));
    cursor.step(K(5));
    EXPECT_TRUE(*cursor == *a[i][K8(1)][K(4)]);
}
