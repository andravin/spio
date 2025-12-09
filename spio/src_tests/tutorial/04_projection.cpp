#include <numeric>

#include "tutorial.h"

// Define tensors A, B, C, and C_tile
/*@spio
[
Tensor("A", dtype.float, Dims(i=16, k=32)),
Tensor("B", dtype.float, Dims(k=32, j=64)),
Tensor("C", dtype.float, Dims(i=16, j=64)),
Tensor("C_tile", dtype.float, Dims(i=8, j=32), Strides(i=64))
]
@spio*/
UTEST(Lesson4, DimensionalProjection) {
    // Create data for matrices a, b, and c.
    A::data_type a_data[A::storage_size()];
    B::data_type b_data[B::storage_size()];
    C::data_type c_data[C::storage_size()];

    // Initialize with counting numbers.
    std::iota(std::begin(a_data), std::end(a_data), 1.0f);
    std::iota(std::begin(b_data), std::end(b_data), 1.0f);
    std::iota(std::begin(c_data), std::end(c_data), 1.0f);

    // Construct matrices a, b, and c.
    auto a = A(a_data);
    auto b = B(b_data);
    auto c = C(c_data);

    // Select coordinates (I, J) for the tiles.
    //
    auto origin = spio::make_coordinates(I(12), J(60));

    // Operations on coordinates use a technique we call dimensional projection:
    // - arithmetic applies to pairs of matching dimensions and passes through others
    // - comparison tests all pairs of matching dimensions
    // - subscript applies matching dimensions and ignores others

    // For matrix a ~ I × K, subscript I matches, and J is ignored.
    auto a_tile = a[origin];

    // For matrix b ~ K × J, subscript J matches, and I is ignored.
    auto b_tile = b[origin];

    // For matrix c ~ I × J, both I and J match.
    auto c_tile = C_tile(c[origin].get());

    // Iterate over the range I(8) × J(32).
    for (auto idx : spio::range(c_tile)) {

        // Iterate over the range K(32).
        for (auto k : spio::range(a.size<K>())) {

            // local and world have dimensions (I, J, K)
            auto local = idx + k;
            auto world = origin + local;

            // Check that world coordinates I and K are less than a's extents.
            // Ignore world coordinate J in the comparison and subscript operations.
            if (world < a.extents()) { EXPECT_TRUE(*a_tile[local] == *a[world]); }

            // Check that world coordinates J and K are less than b's extents.
            // Ignore world coordinate I in the comparison and subscript operations.
            if (world < b.extents()) { EXPECT_TRUE(*b_tile[local] == *b[world]); }
        }

        // Check that world coordinates I and J are less than c's extents.
        if (origin + idx < c.extents()) { EXPECT_TRUE(*c_tile[idx] == *c[origin + idx]); }
    }
}
