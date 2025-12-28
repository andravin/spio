#include "tutorial.h"

// Define dimension types I and J.
//
/*@spio
I = Dim()
J = Dim()
@spio*/

UTEST(Lesson1, TypeSafety) {

    // Dimensions work like integers.
    EXPECT_TRUE(I(2) + I(4) == I(6));
    EXPECT_TRUE(I(8) < I(10));

    // Each dimension is a different CUDA / C++ type.
    static_assert(!std::is_same_v<I, J>, "I and J are different types");

    // Different dimensions cannot be compared. This prevents accidental mixing:
    //
    // EXPECT_EQ(I(5), J(5));
    // error: no match for ‘operator==’ (operand types are ‘I’ and ‘J’)
    //
    // Orthogonal dimensions can be added to produce Coordinates:
    //
    EXPECT_TRUE(I(3) + J(4) == spio::make_coordinates(I(3), J(4)));
}

// Define tensors A and B using dimensions I(16) × K(32) and K(32) × J(64).
//
/*@spio
A = Tensor(dtype.float, Dims(i=16, k=32))
B = Tensor(dtype.float, Dims(k=32, j=64))
@spio*/
UTEST(Lesson1, Commutativity) {

    // Create storage for the matrices.
    A::data_type a_data[A::storage_size()];
    B::data_type b_data[B::storage_size()];

    // Create matrices a and b.
    auto a = A(a_data);
    auto b = B(b_data);

    // Verify matrix sizes.
    EXPECT_TRUE(A::size<I>() == I(16));
    EXPECT_TRUE(A::size<K>() == K(32));
    EXPECT_TRUE(B::size<K>() == K(32));
    EXPECT_TRUE(B::size<J>() == J(64));

    // Define coordinates.
    auto i = I(2);
    auto j = J(3);
    auto k = K(4);

    // Position-free subscripting:
    // Subscript order does not affect the result.
    EXPECT_TRUE(a[i][k].get() == a[k][i].get());
    EXPECT_TRUE(b[k][j].get() == b[j][k].get());

    // Dimensional projection:
    // Coordinates project onto the tensor's supported dimensions.
    auto coords = make_coordinates(i, j, k);
    EXPECT_TRUE(a[coords].get() == a[k][i].get());
    EXPECT_TRUE(b[coords].get() == b[j][k].get());
}
