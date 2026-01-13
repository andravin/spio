#include "tutorial.h"

/*@spio
I = Dim()
J = Dim()
A = Tensor(dtype.float, Dims(I(10), J(10)))
@spio*/

UTEST(Lesson2, RelativeMovement) {
    // Create storage for matrix A.
    A::data_type a_data[A::storage_size()];

    // Create matrix A.
    auto a = A(a_data);

    // Create base cursor at (i=2, j=2).
    auto b = a[I(2)][J(2)];

    // Verify the offset from the base pointer.
    EXPECT_TRUE(b.get() - a_data == 2 * 10 + 2);

    // Move b.
    b.step(I(1));
    b.step(J(1));

    // Verify movement.
    EXPECT_TRUE(b.get() - a_data == 3 * 10 + 3);
}

UTEST(Lesson2, AccumulationLoop) {

    // Create matrix A.
    A::data_type a_data[A::storage_size()];
    auto a = A(a_data);

    // Create cursor at (i=2, j=4).
    auto b = a[I(2)][J(4)];

    for (int step = 0; step < 5; ++step) {
        // Verify the current position.
        EXPECT_TRUE(b.get() == a_data + (2 + step) * 10 + 4);

        // Step by 1 in the I dimension.
        b.step(I(1));
    }
}
