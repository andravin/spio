#include "tutorial.h"

#include <numeric>

/*@spio

# Define the dimensions.
I = Dim()
J = Dim()

# Define the sizes.
i = I(512)
j = J(512)

# Define compound indices for blocks and threads.
BlockIndex = CompoundIndex(Dims(i / 16, j / 16))
ThreadIndex = CompoundIndex(Dims(i % 16, j % 16))

# Define tensor A using dimensions i x j
A = Tensor(dtype.float, Dims(i, j))

# Define aliases for I / 16 and J / 16.
I16 = I / 16
J16 = J / 16

@spio*/
UTEST(Lesson5, CompoundIndex) {

    // Initialize matrix a.
    A::data_type a_data[A::storage_size()];
    std::iota(std::begin(a_data), std::end(a_data), 1.0f);
    auto a = A(a_data);

    // Check the size of the compound indices.
    EXPECT_TRUE(BlockIndex::size() == 32 * 32);
    EXPECT_TRUE(ThreadIndex::size() == 16 * 16);

    // Simulate thread-blocks and threads.
    for (int blockIdx = 0; blockIdx < BlockIndex::size(); ++blockIdx) {
        for (int threadIdx = 0; threadIdx < ThreadIndex::size(); ++threadIdx) {

            // Create a compound index for this block ..
            auto block = BlockIndex(blockIdx);

            // .. and thread.
            auto thread = ThreadIndex(threadIdx);

            // Subscripting with the compound indices ..
            auto b = a[block][thread];

            // .. saves the user from computing the coordinates and offset manually.
            auto block_i16 = blockIdx / 32;
            auto block_j16 = blockIdx % 32;

            auto thread_i = threadIdx / 16;
            auto thread_j = threadIdx % 16;

            auto offset = (block_i16 * 16 + thread_i) * 512 + block_j16 * 16 + thread_j;

            // Check that these two methods are equivalent.
            EXPECT_TRUE(*b == a_data[offset]);
        }
    }
}