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
