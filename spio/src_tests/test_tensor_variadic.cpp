#include "utest.h"
#include "spio/tensor_variadic.h"
#include "spio/dim.h"

using namespace spio;

class HeightDim : public Dim<HeightDim>
{
public:
    using Dim<HeightDim>::Dim;
};

class WidthDim : public Dim<WidthDim>
{
public:
    using Dim<WidthDim>::Dim;
};


template <>
struct utest_type_deducer<HeightDim>
{
    static void _(const HeightDim d)
    {
        UTEST_PRINTF("%d", d.get());
    }
};

template <>
struct utest_type_deducer<WidthDim>
{
    static void _(const WidthDim d)
    {
        UTEST_PRINTF("%d", d.get());
    }
};

// 2D tensor creation helper
template <typename DataType, typename HeightDimType, typename WidthDimType,
          int HeightSize, int WidthSize>
DEVICE constexpr auto make_tensor(DataType *data = nullptr)
{
    return Tensor<
        DataType,
        DimInfo<HeightDimType, HeightSize, WidthSize>, // Height folded by width
        DimInfo<WidthDimType, WidthSize, 1>            // Width with unit stride
        >(data);
}

// Version with custom strides
template <typename DataType, typename HeightDimType, typename WidthDimType,
          int HeightSize, int WidthSize, int HeightStride, int WidthStride>
DEVICE constexpr auto make_tensor_with_strides(DataType *data = nullptr)
{
    return Tensor<
        DataType,
        DimInfo<HeightDimType, HeightSize, HeightStride>,
        DimInfo<WidthDimType, WidthSize, WidthStride>>(data);
}

UTEST(TensorVariadic, tensor_2d)
{
    constexpr HeightDim Height = 480;
    constexpr WidthDim Width = 640;
    constexpr int Size = Height.get() * Width.get();

    float tensor_data[Size];
    for (int i = 0; i < Size; ++i)
    {
        tensor_data[i] = static_cast<float>(i);
    }

    auto tensor = make_tensor<float, HeightDim, WidthDim, Height.get(), Width.get()>(tensor_data);

    EXPECT_EQ(*tensor, tensor_data[0]);
    EXPECT_EQ(*tensor[WidthDim(320)], tensor_data[320]);
    EXPECT_EQ(*tensor[HeightDim(240)], tensor_data[240 * Width.get()]);
    EXPECT_EQ(tensor.get_size<HeightDim>(), Height);

    auto hslice = tensor.slice<240>(HeightDim(20));
    EXPECT_EQ(hslice.get_size<HeightDim>(), HeightDim(240));
    EXPECT_EQ(*hslice, tensor_data[20 * Width.get()]);
    EXPECT_EQ(*hslice[HeightDim(20)][WidthDim(40)], tensor_data[40 * Width.get() + 40]);

    auto wslice = tensor.slice<320>(WidthDim(10));
    EXPECT_EQ(wslice.get_size<WidthDim>(), 320);
    EXPECT_EQ(*wslice, tensor_data[10]);
    EXPECT_EQ(*wslice[HeightDim(20)][WidthDim(10)], tensor_data[20 * Width.get() + 20]);
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
