#include "utest.h"
#include "spio/tensor_variadic.h"
#include "spio/dim.h"

using namespace spio;

namespace
{
    struct HeightDim : public Dim
    {
        using Dim::Dim;
    };

    struct WidthDim : public Dim
    {
        using Dim::Dim;
    };
}

UTEST(TensorVariadic, tensor_2d)
{
    constexpr int Height = 480;
    constexpr int Width = 640;

    float tensor_data[Height * Width];
    for (int i = 0; i < Height * Width; ++i)
    {
        tensor_data[i] = static_cast<float>(i);
    }

    auto tensor = make_tensor<float, HeightDim, WidthDim, Height, Width>(tensor_data);

    EXPECT_EQ(*tensor, tensor_data[0]);
    EXPECT_EQ(*tensor[WidthDim(320)], tensor_data[320]);
    EXPECT_EQ(*tensor[HeightDim(240)], tensor_data[240 * Width]);
    EXPECT_EQ(tensor.get_size<HeightDim>(), Height);

    auto hslice = tensor.slice<240>(HeightDim(20));
    EXPECT_EQ(hslice.get_size<HeightDim>(), 240);
    EXPECT_EQ(*hslice, tensor_data[20 * Width]);
    EXPECT_EQ(*hslice[HeightDim(20)][WidthDim(40)], tensor_data[40 * Width + 40]);

    auto wslice = tensor.slice<320>(WidthDim(10));
    EXPECT_EQ(wslice.get_size<WidthDim>(), 320);
    EXPECT_EQ(*wslice, tensor_data[10]);
    EXPECT_EQ(*wslice[HeightDim(20)][WidthDim(10)], tensor_data[20 * Width + 20]);
}

UTEST(TensorVariadic, tensor_2d_custom_stride)
{
    constexpr int Height = 480;
    constexpr int Width = 640;
    constexpr int Stride = 1024;

    float tensor_data[Height * Stride];
    for (int h = 0; h < Height; ++h)
    {
        for (int w = 0; w < Width; ++w)
        {
            tensor_data[h * Stride + w] = static_cast<float>(h * Width + w);
        }
    }
    
    auto tensor = make_tensor_with_strides<float, HeightDim, WidthDim, Height, Width, Stride, 1>(tensor_data);

    EXPECT_EQ(*tensor, tensor_data[0]);
    EXPECT_EQ(*tensor[WidthDim(320)], tensor_data[320]);
    EXPECT_EQ(*tensor[HeightDim(240)], tensor_data[240 * Stride]);
    EXPECT_EQ(tensor.get_size<HeightDim>(), Height);

    auto hslice = tensor.slice<240>(HeightDim(20));
    EXPECT_EQ(hslice.get_size<HeightDim>(), 240);
    EXPECT_EQ(*hslice, tensor_data[20 * Stride]);
    EXPECT_EQ(*hslice[HeightDim(20)][WidthDim(40)], tensor_data[40 * Stride + 40]);

    auto wslice = tensor.slice<320>(WidthDim(10));
    EXPECT_EQ(wslice.get_size<WidthDim>(), 320);
    EXPECT_EQ(*wslice, tensor_data[10]);
    EXPECT_EQ(*wslice[HeightDim(20)][WidthDim(10)], tensor_data[20 * Stride + 20]);
}

UTEST_MAIN()
