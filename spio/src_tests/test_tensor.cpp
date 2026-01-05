#include "spio/checkerboard_index.h"
#include "dim_test_common.h"
#include "spio/tensor.h"
#include "spio/dim.h"
#include "spio/compound_index.h"

using namespace spio;

TEST_DIM(H);
TEST_DIM(W);
TEST_DIM(I);
TEST_DIM(J);
TEST_DIM(K);
TEST_DIM(M);
TEST_DIM(CHECKERS);

// 2D tensor creation helper
template <typename DataType, typename HeightDimType, typename WidthDimType, int HeightSize,
          int WidthSize>
DEVICE constexpr auto make_tensor(DataType* data = nullptr) {
    return Tensor<DataType, DimInfo<HeightDimType, HeightSize, WidthSize>, // Height folded by width
                  DimInfo<WidthDimType, WidthSize, 1>                      // Width with unit stride
                  >(data);
}

// Version with custom strides
template <typename DataType, typename HeightDimType, typename WidthDimType, int HeightSize,
          int WidthSize, int HeightStride, int WidthStride>
DEVICE constexpr auto make_tensor_with_strides(DataType* data = nullptr) {
    return Tensor<DataType, DimInfo<HeightDimType, HeightSize, HeightStride>,
                  DimInfo<WidthDimType, WidthSize, WidthStride>>(data);
}

UTEST(Tensor, tensor_2d_small) {
    constexpr H Height = 4;
    constexpr W Width = 8;
    constexpr int Size = Height.get() * Width.get();

    float tensor_data[Size];
    for (int i = 0; i < Size; ++i) {
        tensor_data[i] = static_cast<float>(i);
    }

    auto tensor = make_tensor<float, H, W, Height.get(), Width.get()>(tensor_data);
    EXPECT_EQ(tensor.size<H>(), Height);
    EXPECT_EQ(tensor.size<W>(), Width);

    EXPECT_EQ(*tensor, tensor_data[0]);
    EXPECT_EQ(*tensor[W(3)], tensor_data[3]);
    EXPECT_EQ(*tensor[H(2)], tensor_data[2 * Width.get()]);

    EXPECT_EQ(*tensor[W(3)].rebase(), tensor_data[3]);
    EXPECT_EQ(*tensor[H(2)].rebase(), tensor_data[2 * Width.get()]);
    EXPECT_EQ(*tensor[W(3)].rebase()[H(2)], tensor_data[3 + 2 * Width.get()]);
}

UTEST(Tensor, tensor_2d) {
    constexpr H Height = 480;
    constexpr W Width = 640;
    constexpr int Size = Height.get() * Width.get();

    float tensor_data[Size];
    for (int i = 0; i < Size; ++i) {
        tensor_data[i] = static_cast<float>(i);
    }

    auto tensor = make_tensor<float, H, W, Height.get(), Width.get()>(tensor_data);

    EXPECT_EQ(*tensor, tensor_data[0]);
    EXPECT_EQ(*tensor[W(320)], tensor_data[320]);
    EXPECT_EQ(*tensor[H(240)], tensor_data[240 * Width.get()]);
    EXPECT_EQ(tensor.size<H>(), Height);

    auto hslice = tensor.slice<240>(H(20));
    EXPECT_EQ(hslice.size<H>(), H(240));
    EXPECT_EQ(*hslice, tensor_data[20 * Width.get()]);
    EXPECT_EQ(*hslice[H(20)][W(40)], tensor_data[40 * Width.get() + 40]);

    auto wslice = tensor.slice<320>(W(10));
    EXPECT_EQ(wslice.size<W>(), 320);
    EXPECT_EQ(*wslice, tensor_data[10]);
    EXPECT_EQ(*wslice[H(20)][W(10)], tensor_data[20 * Width.get() + 20]);
}

UTEST(Tensor, tensor_2d_custom_stride) {
    constexpr int Height = 480;
    constexpr int Width = 640;
    constexpr int Stride = 1024;

    float tensor_data[Height * Stride];
    for (int h = 0; h < Height; ++h) {
        for (int w = 0; w < Width; ++w) {
            tensor_data[h * Stride + w] = static_cast<float>(h * Width + w);
        }
    }

    auto tensor = make_tensor_with_strides<float, H, W, Height, Width, Stride, 1>(tensor_data);

    for (auto h : range(tensor.size<H>())) {
        for (auto w : range(tensor.size<W>())) {
            EXPECT_EQ(*tensor[h][w], tensor_data[h.get() * Stride + w.get()]);
        }
    }

    EXPECT_EQ(*tensor, tensor_data[0]);
    EXPECT_EQ(*tensor[W(320)], tensor_data[320]);
    EXPECT_EQ(*tensor[H(240)], tensor_data[240 * Stride]);
    EXPECT_EQ(tensor.size<H>(), Height);

    auto hslice = tensor.slice<240>(H(20));
    EXPECT_EQ(hslice.size<H>(), 240);
    EXPECT_EQ(*hslice, tensor_data[20 * Stride]);
    EXPECT_EQ(*hslice[H(20)][W(40)], tensor_data[40 * Stride + 40]);

    auto wslice = tensor.slice<320>(W(10));
    EXPECT_EQ(wslice.size<W>(), 320);
    EXPECT_EQ(*wslice, tensor_data[10]);
    EXPECT_EQ(*wslice[H(20)][W(10)], tensor_data[20 * Stride + 20]);
}

// Test the tensor[index] subscript operator
UTEST(Tensor, IndexSubscript) {
    // 2D Test
    // Create a 3x4 tensor
    constexpr int HEIGHT = 3;
    constexpr int WIDTH = 4;
    float data2d[HEIGHT * WIDTH] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    // Create a 2D tensor
    auto tensor2d = Tensor<float, DimInfo<I, HEIGHT, WIDTH>, DimInfo<J, WIDTH, 1>>(data2d);

    // Create a matching CompoundIndex type
    using Idx2D = CompoundIndex<DimInfo<I, HEIGHT, WIDTH>, DimInfo<J, WIDTH, 1>>;

    for (int offset = 0; offset < HEIGHT * WIDTH; ++offset) {
        // Create index from coordinates
        auto idx = Idx2D(offset);

        // Get data via traditional subscript
        float val1 = *tensor2d[idx.get<I>()][idx.get<J>()];

        // Get data via index subscript
        float val2 = *tensor2d[idx];

        // Values should be identical
        EXPECT_EQ(val1, val2);
        EXPECT_EQ(val1, idx.get<I>().get() * WIDTH + idx.get<J>().get());
    }

    // 3D Test
    // Create a 2x3x4 tensor
    constexpr int DEPTH = 2;
    float data3d[DEPTH * HEIGHT * WIDTH] = {// Layer 0
                                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                            // Layer 1
                                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};

    // Create a 3D tensor
    auto tensor3d = Tensor<float, DimInfo<I, DEPTH, HEIGHT * WIDTH>, DimInfo<J, HEIGHT, WIDTH>,
                           DimInfo<K, WIDTH, 1>>(data3d);

    // Create a matching CompoundIndex type
    using Idx3D = CompoundIndex<DimInfo<I, DEPTH, HEIGHT * WIDTH>, DimInfo<J, HEIGHT, WIDTH>,
                                DimInfo<K, WIDTH, 1>>;

    for (int offset = 0; offset < DEPTH * HEIGHT * WIDTH; ++offset) {
        // Create index from coordinates
        auto idx = Idx3D(offset);

        // Get data via traditional subscript
        float val1 = *tensor3d[idx.get<I>()][idx.get<J>()][idx.get<K>()];

        // Get data via index subscript
        float val2 = *tensor3d[idx];

        // Values should be identical
        EXPECT_EQ(val1, val2);
    }
}

UTEST(Tensor, tensor_extent_simple) {
    // Simple 2D tensor without folds
    constexpr int Height = 16;
    constexpr int Width = 32;

    auto tensor = Tensor<float, DimInfo<H, Height, Width>, DimInfo<W, Width, 1>>();

    // extent<H>() should return the total H extent
    EXPECT_EQ(tensor.extent<H>(), H(Height));
    EXPECT_EQ(tensor.extent<W>(), W(Width));
}

UTEST(Tensor, tensor_extent_folded) {
    // Tensor with folded dimensions
    constexpr int Height = 16;
    constexpr int Width = 16;
    constexpr int FoldSize = 4;

    using FoldedH = Fold<H, FoldSize>;
    using FoldedW = Fold<W, FoldSize>;

    auto tensor = Tensor<float, DimInfo<FoldedH, Height / FoldSize, Width>,
                         DimInfo<FoldedW, Width / FoldSize, 1>>();

    // size<FoldedH>() returns the literal dimension size
    EXPECT_EQ(tensor.size<FoldedH>(), FoldedH(Height / FoldSize));
    EXPECT_EQ(tensor.size<FoldedW>(), FoldedW(Width / FoldSize));

    // extent<H>() returns the total extent in base H units
    EXPECT_EQ(tensor.extent<H>(), H(Height));
    EXPECT_EQ(tensor.extent<W>(), W(Width));

    // extent<FoldedH>() returns the total extent converted to FoldedH
    EXPECT_EQ(tensor.extent<FoldedH>(), FoldedH(Height / FoldSize));
    EXPECT_EQ(tensor.extent<FoldedW>(), FoldedW(Width / FoldSize));
}

UTEST(Tensor, tensor_extent_multiple_folds) {
    // Tensor with multiple folds of the same base dimension (hierarchical)
    // K8 (coarse) and K (fine) are complementary folds of dimension K

    using K8 = Fold<K, 8>;

    constexpr int k8_size = 8; // 8 groups of 8
    constexpr int k_size = 8;  // 8 elements per group
    // Total K extent = 8 * 8 = 64

    auto tensor = Tensor<float, DimInfo<K8, k8_size, k_size>, DimInfo<K, k_size, 1>>();

    // size<K8>() and size<K>() return literal dimension sizes
    EXPECT_EQ(tensor.size<K8>(), K8(k8_size));
    EXPECT_EQ(tensor.size<K>(), K(k_size));

    // extent<K>() returns the total K extent from the coarsest fold
    // Coarsest is K8 with stride 8, size 8 -> total = 8 * 8 = 64
    EXPECT_EQ(tensor.extent<K>(), K(64));

    // extent<K8>() returns the total extent converted to K8 units
    // 64 base units / 8 stride = 8
    EXPECT_EQ(tensor.extent<K8>(), K8(8));
}

UTEST(Tensor, tensor_extent_different_fold_levels) {
    // Test querying extent at different fold levels

    using M4 = Fold<M, 4>;
    using M16 = Fold<M, 16>;

    // Tensor with M16 dimension only (size 4, stride 16 -> total M = 64)
    auto tensor = Tensor<float, DimInfo<M16, 4, 1>>();

    EXPECT_EQ(tensor.size<M16>(), M16(4));

    // extent<M>() should return M(64)
    EXPECT_EQ(tensor.extent<M>(), M(64));

    // extent<M4>() should return M4(16) since 64 / 4 = 16
    EXPECT_EQ(tensor.extent<M4>(), M4(16));

    // extent<M16>() should return M16(4) since 64 / 16 = 4
    EXPECT_EQ(tensor.extent<M16>(), M16(4));
}

UTEST(Cursor, cursor_size_simple) {
    // Test Cursor::size() with simple dimensions
    constexpr int Height = 16;
    constexpr int Width = 32;

    using T = Tensor<float, DimInfo<H, Height, Width>, DimInfo<W, Width, 1>>;
    float data[T::storage_size()];
    auto tensor = T(data);

    // Get a cursor by subscripting
    auto cursor = tensor[H(0)];

    // Cursor should have same size as tensor for each dimension
    EXPECT_EQ(cursor.size<H>(), H(Height));
    EXPECT_EQ(cursor.size<W>(), W(Width));
}

UTEST(Cursor, cursor_size_folded) {
    // Test Cursor::size() with folded dimensions
    using K8 = Fold<K, 8>;

    constexpr int k8_size = 8;
    constexpr int k_size = 8;

    using T = Tensor<float, DimInfo<K8, k8_size, k_size>, DimInfo<K, k_size, 1>>;
    float data[T::storage_size()];
    auto tensor = T(data);

    auto cursor = tensor[K(0)];

    // Cursor should report correct sizes for each dimension
    EXPECT_EQ(cursor.size<K8>(), K8(k8_size));
    EXPECT_EQ(cursor.size<K>(), K(k_size));
}

UTEST(Cursor, cursor_extent_matches_tensor) {
    // Verify Cursor::extent() returns same values as Tensor::extent()
    using K8 = Fold<K, 8>;

    constexpr int k8_size = 8;
    constexpr int k_size = 8;
    constexpr int m_size = 16;

    using T = Tensor<float, DimInfo<K8, k8_size, m_size * k_size>, DimInfo<M, m_size, k_size>,
                     DimInfo<K, k_size, 1>>;

    float data[T::storage_size()];
    auto tensor = T(data);
    auto cursor = tensor[M(5)];

    // Cursor extent should match tensor extent
    EXPECT_EQ(cursor.extent<K>(), tensor.extent<K>());
    EXPECT_EQ(cursor.extent<K8>(), tensor.extent<K8>());
    EXPECT_EQ(cursor.extent<M>(), tensor.extent<M>());

    // Verify actual values
    EXPECT_EQ(cursor.extent<K>(), K(64));  // 8 * 8 = 64
    EXPECT_EQ(cursor.extent<K8>(), K8(8)); // coarsest K8 size
    EXPECT_EQ(cursor.extent<M>(), M(16));
}

UTEST(Cursor, cursor_extent_different_fold_levels) {
    // Test Cursor::extent() at different fold levels (mirrors tensor test)
    using M4 = Fold<M, 4>;
    using M16 = Fold<M, 16>;

    using T = Tensor<float, DimInfo<M16, 4, 1>>;
    float data[T::storage_size()];
    auto tensor = T(data);
    auto cursor = tensor[M16(0)];

    EXPECT_EQ(cursor.size<M16>(), M16(4));

    // extent<M>() should return M(64)
    EXPECT_EQ(cursor.extent<M>(), M(64));

    // extent<M4>() should return M4(16) since 64 / 4 = 16
    EXPECT_EQ(cursor.extent<M4>(), M4(16));

    // extent<M16>() should return M16(4) since 64 / 16 = 4
    EXPECT_EQ(cursor.extent<M16>(), M16(4));
}

UTEST(Cursor, cursor_type_alias_extent) {
    // Test using Cursor type alias for extent (common pattern in kernels)
    using K8 = Fold<K, 8>;

    constexpr int k8_size = 8;
    constexpr int k_size = 8;

    using T = Tensor<float, DimInfo<K8, k8_size, k_size>, DimInfo<K, k_size, 1>>;
    float data[T::storage_size()];
    auto tensor = T(data);

    auto cursor = tensor[K(0)];
    using CursorType = decltype(cursor);

    // Static method call on type alias (kernel pattern)
    constexpr auto extent_k = CursorType::extent<K>();
    EXPECT_EQ(extent_k, K(64));

    constexpr auto extent_k8 = CursorType::extent<K8>();
    EXPECT_EQ(extent_k8, K8(8));
}

UTEST(Tensor, subscript_projection_onto_folds) {
    // Test subscripting a tensor with multiple folds of the same base dimension
    // Ensure that projection of a dimension ontofolds works correctly

    using K8 = Fold<K, 8>;

    constexpr int k8_size = 4;
    constexpr int k_size = 8;
    constexpr int m_size = 16;
    constexpr int total_k_size = k8_size * 8;

    using T = Tensor<float, DimInfo<K8, k8_size, m_size * k_size>, DimInfo<M, m_size, k_size>,
                     DimInfo<K, k_size, 1>>;

    float data[T::total_size];
    for (int i = 0; i < T::total_size; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto tensor = T(data);

    EXPECT_EQ(tensor.extent<K>(), K(k8_size * 8));

    for (int m = 0; m < m_size; ++m) {
        for (int k = 0; k < total_k_size; ++k) {
            int k8_index = k / 8;
            int k_index = k % 8;
            // This test works because all the dimensions are independent.
            EXPECT_EQ(*tensor[K8(k8_index)][K(k_index)][M(m)], *tensor[K(k)][M(m)]);
        }
    }
}

UTEST(Tensor, subscript_multiple_folds_with_carry) {
    // Test subscripting a tensor with multiple folds of the same base dimension
    // Ensure that carry-over between folds works correctly

    using K8 = Fold<K, 8>;

    // Total K extent = 4 * 8 = 32
    constexpr int k8_size = 4; // 4 groups of 8
    constexpr int k_size = 8;  // 8 elements per group
    constexpr int m_size = 16;

    using T = Tensor<float, DimInfo<K8, k8_size, m_size * k_size>, DimInfo<M, m_size, k_size>,
                     DimInfo<K, k_size, 1>>;

    float data[T::total_size];
    for (int i = 0; i < T::total_size; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto tensor = T(data);

    // This test should now pass with deferred folding:
    // K(4) + K(4) = K(8), which should map to K8=1, K=0
    EXPECT_EQ(*tensor[K(4)][K(4)], *tensor[K(8)]);
}

UTEST(Tensor, deferred_folding_equivalence) {
    // Test that a[e][f][g] == a[e + f + g] with deferred folding

    using K8 = Fold<K, 8>;

    constexpr int k8_size = 4;
    constexpr int k_size = 8;
    constexpr int m_size = 16;

    using T = Tensor<float, DimInfo<K8, k8_size, m_size * k_size>, DimInfo<M, m_size, k_size>,
                     DimInfo<K, k_size, 1>>;

    float data[T::total_size];
    for (int i = 0; i < T::total_size; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto tensor = T(data);

    // Test various combinations
    for (int e = 0; e < 8; ++e) {
        for (int f = 0; f < 8; ++f) {
            for (int g = 0; g < 8; ++g) {
                // Subscripting separately vs combined should give same result
                EXPECT_EQ(*tensor[K(e)][K(f)][K(g)], *tensor[K(e + f + g)]);
            }
        }
    }

    // Test with M dimension as well
    for (int k = 0; k < 16; ++k) {
        for (int m = 0; m < m_size; ++m) {
            // Different orderings should be equivalent
            EXPECT_EQ(*tensor[K(k)][M(m)], *tensor[M(m)][K(k)]);
        }
    }
}

UTEST(Tensor, deferred_folding_with_coordinates) {
    // Test that coordinates addition works correctly with deferred folding

    using K8 = Fold<K, 8>;

    constexpr int k8_size = 4;
    constexpr int k_size = 8;
    constexpr int m_size = 16;

    using T = Tensor<float, DimInfo<K8, k8_size, m_size * k_size>, DimInfo<M, m_size, k_size>,
                     DimInfo<K, k_size, 1>>;

    float data[T::total_size];
    for (int i = 0; i < T::total_size; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto tensor = T(data);

    // Test that coordinates work with the tensor
    auto coord1 = make_coordinates(K(5), M(3));
    auto coord2 = make_coordinates(K(7), M(2));

    // Subscripting with coordinates should work
    EXPECT_EQ(*tensor[coord1], *tensor[K(5)][M(3)]);
    EXPECT_EQ(*tensor[coord2], *tensor[K(7)][M(2)]);

    // Combined coordinates should equal sequential subscripts
    auto combined = coord1 + coord2;
    EXPECT_EQ(*tensor[combined], *tensor[K(12)][M(5)]);
}

UTEST(Tensor, deferred_folding_with_step) {
    // Test that step() correctly handles cross-fold carry behavior

    using K8 = Fold<K, 8>;

    constexpr int k8_size = 4;
    constexpr int k_size = 8;
    constexpr int m_size = 16;

    using T = Tensor<float, DimInfo<K8, k8_size, m_size * k_size>, DimInfo<M, m_size, k_size>,
                     DimInfo<K, k_size, 1>>;

    float data[T::total_size];
    for (int i = 0; i < T::total_size; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto tensor = T(data);

    // Test stepping by K(4) repeatedly
    // Each step of K(4) should accumulate, and cross-fold carry should work correctly
    auto cursor = tensor[M(3)];

    for (int step_count = 0; step_count < 8; ++step_count) {
        int expected_k = step_count * 4;
        // Verify cursor points to correct element
        EXPECT_EQ(*cursor, *tensor[K(expected_k)][M(3)]);

        // Step by K(4)
        cursor.step(K(4));
    }

    // Test stepping by K(1) to verify fine-grained carry
    cursor = tensor[M(5)];

    for (int step_count = 0; step_count < 20; ++step_count) {
        EXPECT_EQ(*cursor, *tensor[K(step_count)][M(5)]);
        cursor.step(K(1));
    }

    // Test stepping by K(8) - exactly one K8 unit
    cursor = tensor[M(7)];

    for (int step_count = 0; step_count < 4; ++step_count) {
        int expected_k = step_count * 8;
        EXPECT_EQ(*cursor, *tensor[K(expected_k)][M(7)]);
        cursor.step(K(8));
    }

    // Test stepping by K8(1) directly
    cursor = tensor[M(2)];

    for (int step_count = 0; step_count < 4; ++step_count) {
        int expected_k = step_count * 8;
        EXPECT_EQ(*cursor, *tensor[K(expected_k)][M(2)]);
        cursor.step(K8(1));
    }

    // Test mixed stepping - alternate between K(3) and K(5)
    cursor = tensor[M(1)];
    int accumulated_k = 0;

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(*cursor, *tensor[K(accumulated_k)][M(1)]);
        cursor.step(K(3));
        accumulated_k += 3;

        EXPECT_EQ(*cursor, *tensor[K(accumulated_k)][M(1)]);
        cursor.step(K(5));
        accumulated_k += 5;
    }
}

UTEST(Tensor, checkerboardindex_derived_dim_subscript) {
    // CheckerboardIndex<8, I, K8, CHECKERS, 16> will have input_dims: I(8), K8(2), output_dim:
    // CHECKERS(16)
    using K8 = Fold<K, 8>;
    constexpr int Ranks = 8;
    constexpr int Size = 16;
    using Checkerboard = CheckerboardIndex<Ranks, I, K8, CHECKERS, Size>;

    // Tensor with derived dimension AND its output DimInfo
    // The derived dim's output_dims is DimSize<CHECKERS, 16>, so we need DimInfo<CHECKERS, 16, 1>
    float data[Size];
    for (int i = 0; i < Size; ++i)
        data[i] = static_cast<float>(i);
    Tensor<float, Checkerboard, DimInfo<CHECKERS, Size, 1>> tensor(data);

    // Test subscript with input_dims (PairDim, ColorDim)
    for (int idx = 0; idx < Size; ++idx) {
        auto i = idx / 2;
        auto k8 = idx % 2;
        auto coords = make_coordinates(I(i), K8(k8));

        // Compute expected offset using CheckerboardIndex logic
        Checkerboard expected_idx{I(i), K8(k8)};
        int expected_offset = expected_idx.offset().get();
        EXPECT_EQ(*tensor[coords], data[expected_offset]);
    }
}

UTEST(Tensor, checkerboardindex_derived_dim_subscript_with_coarse_fold) {
    using K8 = Fold<K, 8>;
    using K64 = Fold<K, 64>;
    constexpr int Ranks = 8;
    constexpr int Size = 16;
    using Checkerboard = CheckerboardIndex<Ranks, I, K8, CHECKERS, Size>;

    using MyTensor = Tensor<float, Checkerboard, DimInfo<K64, 4, Size>, DimInfo<CHECKERS, Size, 1>>;
    MyTensor::data_type data[MyTensor::storage_size()];
    MyTensor tensor(data);
    for (int i = 0; i < MyTensor::storage_size(); ++i)
        data[i] = static_cast<float>(i);

    for (int k64 = 0; k64 < 4; ++k64) {
        for (int idx = 0; idx < Size; ++idx) {
            auto i = idx / 2;
            auto k8 = idx % 2;
            auto coords = make_coordinates(K64(k64), I(i), K8(k8));

            // Compute expected offset using CheckerboardIndex logic
            Checkerboard expected_idx{I(i), K8(k8)};
            int expected_offset = expected_idx.offset().get();
            EXPECT_EQ(*tensor[coords], data[k64 * 16 + expected_offset]);
        }
    }

    for (int k8 = 0; k8 < 4 * (64 / 8); ++k8) {
        for (int i = 0; i < 8; ++i) {
            auto coords = make_coordinates(I(i), K8(k8));

            // Compute expected offset using CheckerboardIndex logic
            Checkerboard expected_idx{I(i), K8(k8 % 2)};
            int expected_offset = expected_idx.offset().get();
            auto k64 = k8 / 8;
            EXPECT_EQ(*tensor[coords], data[k64 * 16 + expected_offset]);
        }
    }
}

UTEST(Tensor, checkerboardindex_derived_dim_subscript_with_static_dim_mixing) {
    using K8 = Fold<K, 8>;
    using K64 = Fold<K, 64>;
    using I16 = Fold<I, 16>;
    constexpr int Ranks = 8;
    constexpr int Size = 32;
    using Checkerboard = CheckerboardIndex<Ranks, I, K8, CHECKERS, Size>;

    using MyTensor = Tensor<float, Checkerboard, DimInfo<I16, 4, Size * 4>, DimInfo<K64, 4, Size>,
                            DimInfo<CHECKERS, Size, 1>>;
    float data[MyTensor::storage_size()];
    MyTensor tensor(data);
    for (int i = 0; i < MyTensor::storage_size(); ++i)
        data[i] = static_cast<float>(i);

    for (int k8 = 0; k8 < 4 * (64 / 8); ++k8) {
        for (int i = 0; i < 64; ++i) {
            auto coords = make_coordinates(I(i), K8(k8));

            // Compute expected offset using CheckerboardIndex logic
            Checkerboard expected_idx{I(i % 16), K8(k8 % 2)};
            int expected_offset = expected_idx.offset().get();
            auto k64 = k8 / 8;
            auto i16 = i / 16;
            EXPECT_EQ(*tensor[coords], data[i16 * 4 * Size + k64 * Size + expected_offset]);
        }
    }
}

// Test using coarse folds to subscript a tensor with both fine folds and derived dimensions
// This mirrors the kernel scenario: WARP_J(0 or 1) subscripting a tensor with J16 and Checkerboard
UTEST(Tensor, checkerboardindex_with_coarse_fold_subscript) {
    using K8 = Fold<K, 8>;
    using J16 = Fold<J, 16>;
    using J64 = Fold<J, 64>; // Like WARP_J in the kernel
    constexpr int Ranks = 8;
    constexpr int Size = 32;
    using Checkerboard = CheckerboardIndex<Ranks, J, K8, CHECKERS, Size>;

    // Tensor layout mirrors BSmem: J16 dimension (tiles) + Checkerboard (swizzled storage)
    // J16 has 8 tiles to cover j64=0 and j64=1 (each j64 spans 4 J16 tiles)
    using MyTensor = Tensor<float, Checkerboard, DimInfo<J16, 8, Size>, DimInfo<CHECKERS, Size, 1>>;

    float data[MyTensor::storage_size()];
    MyTensor tensor(data);
    for (int i = 0; i < MyTensor::storage_size(); ++i)
        data[i] = static_cast<float>(i);

    // Debug: print storage size and first failures
    int failure_count = 0;
    constexpr int max_failures = 5;

    // Test 1: Subscript with J64 (like WARP_J(0) and WARP_J(1))
    for (int j64 = 0; j64 < 2; ++j64) {
        for (int j = 0; j < 16; ++j) {
            for (int k8 = 0; k8 < 2; ++k8) {
                auto coarse_coords = make_coordinates(J64(j64));
                auto fine_coords = make_coordinates(J(j), K8(k8));

                int j16_idx = j64 * 4 + j / 16;
                Checkerboard expected_idx{J(j % 16), K8(k8)};
                int expected_offset = expected_idx.offset().get();
                int expected_data_idx = j16_idx * Size + expected_offset;

                auto cursor1 = tensor[coarse_coords];
                auto cursor2 = cursor1[fine_coords];

                // Debug: compute actual pointer offset
                int actual_offset = static_cast<int>(cursor2.get() - data);

                if (*cursor2 != data[expected_data_idx] && failure_count < max_failures) {
                    printf("FAIL: j64=%d j=%d k8=%d | expected_idx=%d actual_offset=%d | "
                           "j16_idx=%d checkerboard_offset=%d\n",
                           j64, j, k8, expected_data_idx, actual_offset, j16_idx, expected_offset);
                    printf("  cursor1.coords: J16=%d\n", cursor1.coordinates().get<J16>().get());
                    printf("  cursor2.coords: J16=%d CHECKERS=%d\n",
                           cursor2.coordinates().get<J16>().get(),
                           cursor2.coordinates().get<CHECKERS>().get());
                    failure_count++;
                }
                EXPECT_EQ(*cursor2, data[expected_data_idx]);
            }
        }
    }
}

// Test dimensional projection: subscript with coordinates containing extra dimensions
// that don't exist in the tensor. The tensor should ignore non-matching dimensions.
UTEST(Tensor, dimensional_projection_ignores_extra_dims) {
    // Tensor A has dimensions I × K (like a matrix)
    using A = Tensor<float, DimInfo<I, 16, 1>, DimInfo<K, 32, 16>>;

    float data[A::storage_size()];
    for (size_t i = 0; i < A::storage_size(); ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    auto a = A(data);

    // Create coordinates with I, J, and K - but tensor A only has I and K
    // J should be ignored when subscripting
    auto origin = make_coordinates(I(5), J(99)); // J=99 should be ignored

    // a[origin] should only apply I(5), ignoring J
    auto a_tile = a[origin];

    // Verify by comparing with explicit subscript
    EXPECT_EQ(*a_tile[K(7)], *a[I(5)][K(7)]);

    // Also test with coordinates that have all three dims
    auto world = make_coordinates(I(3), J(42), K(10));
    // Should match a[I(3)][K(10)], ignoring J
    EXPECT_EQ(*a[world], *a[I(3)][K(10)]);
}

// Test the exact pattern from test_mma_checkerboard.py:
// AGlobal = Tensor(dtype.half8, Dims(i16=i16, k16=k16, i=-1, k8=-1))[ALoadGlobalIndex, BlockIdx]
//
// This creates a tensor with dimensions I16 x K16 x I x K8, then subscripts it with:
// 1. ALoadGlobalIndex: CompoundIndex(Dims(i=block_m, k8=2)) - thread-local offset
// 2. BlockIdx: CompoundIndex(Dims(block_i_wave, block_j, block_i_local)) - block offset
//
// The key challenge is that BlockIdx has TWO folds of I (block_i_wave and block_i_local)
// that must BOTH contribute to the tensor's I16 and I dimensions.
// Also tests that BLOCK_J (a fold of J) is correctly ignored since the tensor has no J dimension.
UTEST(Tensor, double_subscript_with_wave_block_idx) {
    using I16 = Fold<I, 16>;
    using K16 = Fold<K, 16>;
    using K8 = Fold<K, 8>;

    // Tensor similar to AGlobal: I16 x K16 x I x K8
    // Sizes: i16=4, k16=2, i=16, k8=2 (total 4*2*16*2 = 256)
    constexpr int i16_size = 4;
    constexpr int k16_size = 2;
    constexpr int i_size = 16;
    constexpr int k8_size = 2;

    // Strides for row-major layout: i16 is outermost, k8 is innermost
    constexpr int k8_stride = 1;
    constexpr int i_stride = k8_size * k8_stride;     // 2
    constexpr int k16_stride = i_size * i_stride;     // 32
    constexpr int i16_stride = k16_size * k16_stride; // 64

    using ATensor =
        Tensor<float, DimInfo<I16, i16_size, i16_stride>, DimInfo<K16, k16_size, k16_stride>,
               DimInfo<I, i_size, i_stride>, DimInfo<K8, k8_size, k8_stride>>;

    constexpr int total_size = i16_size * i16_stride;

    float data[total_size];
    for (int i = 0; i < total_size; ++i) {
        data[i] = static_cast<float>(i);
    }
    auto tensor = ATensor(data);

    // Wave-based BlockIdx: Dims(block_i_wave=2, block_j=4, block_i_local=2)
    // This simulates: wave_size=2, blocks_m=4, blocks_n=4
    // block_i_wave has stride wave_stride = wave_size * block_m (e.g., 2 * 32 = 64)
    // block_i_local has stride block_m (e.g., 32)
    // block_j has stride block_n (e.g., 16)
    //
    // For this test, use:
    // - block_i_wave: Fold<I, 32> (2 blocks of 32 each = 64 total I)
    // - block_i_local: Fold<I, 16> (2 blocks of 16 each within wave)
    // - block_j: Fold<J, 8> (4 blocks of 8 each = 32 total J, ignored by tensor)
    //
    // Total I extent = 2 * 32 = 64 = 4 * 16 (matches i16_size * i_size)

    using BLOCK_I_WAVE = Fold<I, 32>;  // wave_stride = 32
    using BLOCK_I_LOCAL = Fold<I, 16>; // block_m = 16
    using BLOCK_J = Fold<J, 8>;        // block_n = 8 (ignored by tensor)

    // BlockIdx: block_i_wave=2, block_j=4, block_i_local=2
    // Total size = 2 * 4 * 2 = 16
    constexpr int block_i_wave_size = 2;
    constexpr int block_j_size = 4;
    constexpr int block_i_local_size = 2;

    // Strides for the compound index (innermost to outermost: local, j, wave)
    constexpr int block_i_local_stride = 1;
    constexpr int block_j_stride = block_i_local_size;
    constexpr int block_i_wave_stride = block_j_size * block_j_stride;

    using BlockIdx =
        CompoundIndex<DimInfo<BLOCK_I_WAVE, block_i_wave_size, block_i_wave_stride>,
                      DimInfo<BLOCK_J, block_j_size, block_j_stride>,
                      DimInfo<BLOCK_I_LOCAL, block_i_local_size, block_i_local_stride>>;

    // Test: for each block index, verify that tensor[BlockIdx] gives correct offset
    for (int block_idx = 0; block_idx < BlockIdx::size(); ++block_idx) {
        auto idx = BlockIdx(block_idx);

        auto block_i_wave = idx.get<BLOCK_I_WAVE>();
        auto block_j = idx.get<BLOCK_J>();
        auto block_i_local = idx.get<BLOCK_I_LOCAL>();

        // Compute the expected I coordinate from the two I folds
        // I = block_i_wave * 32 + block_i_local * 16
        int expected_i = block_i_wave.get() * 32 + block_i_local.get() * 16;

        // Convert to I16 and I components for tensor access
        int expected_i16 = expected_i / 16;
        int expected_i_inner = expected_i % 16;

        // Access tensor with the compound index - should apply I folds correctly
        auto cursor = tensor[idx];

        // Compare with explicit subscript using computed I16 and I values
        // (at K16=0, K8=0 for simplicity)
        auto expected_cursor = tensor[I16(expected_i16)][I(expected_i_inner)];

        // The pointers should match
        EXPECT_EQ(cursor.get(), expected_cursor.get());
    }
}

// Test chained subscripts: verify that order of subscript application doesn't matter
// for independent dimensions, but accumulates correctly for shared base dimensions.
UTEST(Tensor, double_subscript_order_independence) {
    using I16 = Fold<I, 16>;
    using K8 = Fold<K, 8>;

    // Simple tensor with I16 x I x K8 layout
    constexpr int i16_size = 4;
    constexpr int i_size = 16;
    constexpr int k8_size = 4;

    constexpr int k8_stride = 1;
    constexpr int i_stride = k8_size;
    constexpr int i16_stride = i_size * i_stride;

    using TestTensor = Tensor<float, DimInfo<I16, i16_size, i16_stride>,
                              DimInfo<I, i_size, i_stride>, DimInfo<K8, k8_size, k8_stride>>;

    float data[TestTensor::storage_size()];
    for (size_t i = 0; i < TestTensor::storage_size(); ++i) {
        data[i] = static_cast<float>(i);
    }
    auto tensor = TestTensor(data);

    // Two indices with overlapping I dimension
    using BLOCK_I = Fold<I, 32>; // Block-level I offset
    using THREAD_I = Fold<I, 1>; // Thread-level I offset (just I)

    // Create indices
    using BlockIndex = CompoundIndex<DimInfo<BLOCK_I, 2, 1>>; // 2 blocks of 32
    using ThreadIndex = CompoundIndex<DimInfo<I, 16, 4>, DimInfo<K8, 4, 1>>;

    // Test: block_i=1, thread_i=5, k8=2
    // Total I = 1*32 + 5 = 37
    // Expected: I16 = 37/16 = 2, I = 37%16 = 5
    auto block_idx = BlockIndex(1);           // BLOCK_I = 1
    auto thread_idx = ThreadIndex(5 * 4 + 2); // I = 5, K8 = 2

    // Order 1: tensor[thread_idx][block_idx]
    auto cursor1 = tensor[thread_idx][block_idx];

    // Order 2: tensor[block_idx][thread_idx]
    auto cursor2 = tensor[block_idx][thread_idx];

    // Both should give same result since dimensions accumulate
    EXPECT_EQ(cursor1.get(), cursor2.get());

    // Verify the actual offset
    // I = 37, K8 = 2
    // I16 = 2, I_inner = 5
    // offset = 2 * 64 + 5 * 4 + 2 = 128 + 20 + 2 = 150
    int expected_offset = 2 * i16_stride + 5 * i_stride + 2 * k8_stride;
    EXPECT_EQ(static_cast<int>(cursor1.get() - data), expected_offset);
}
