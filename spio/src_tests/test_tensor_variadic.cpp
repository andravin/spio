#include "utest.h"
#include "spio/tensor_variadic.h"
#include "spio/dim.h"
#include "spio/index_variadic.h"

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

// Test the tensor[index] subscript operator
UTEST(TensorVariadic, IndexSubscript) {
    // 2D Test
    // Create a 3x4 tensor
    constexpr int HEIGHT = 3;
    constexpr int WIDTH = 4;
    float data2d[HEIGHT * WIDTH] = {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11
    };
    
    // Define dimension types
    class I_Dim : public Dim<I_Dim> { public: using Dim<I_Dim>::Dim; };
    class J_Dim : public Dim<J_Dim> { public: using Dim<J_Dim>::Dim; };
    class K_Dim : public Dim<K_Dim> { public: using Dim<K_Dim>::Dim; };
    
    // Create a 2D tensor
    auto tensor2d = Tensor<float, 
        DimInfo<I_Dim, HEIGHT, WIDTH>, 
        DimInfo<J_Dim, WIDTH, 1>
    >(data2d);
    
    // Create a matching Index type
    using Idx2D = Index<
        DimInfo<I_Dim, HEIGHT, WIDTH>, 
        DimInfo<J_Dim, WIDTH, 1>
    >;
    
    // Test that tensor[idx] gives the same result as tensor[i][j]
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            // Create index from coordinates
            auto idx = Idx2D::from_coords(I_Dim(i), J_Dim(j));
            
            // Get data via traditional subscript
            float val1 = *tensor2d[I_Dim(i)][J_Dim(j)];
            
            // Get data via index subscript
            float val2 = *tensor2d[idx];
            
            // Values should be identical
            EXPECT_EQ(val1, val2);
            EXPECT_EQ(val1, i * WIDTH + j);
        }
    }
    
    // 3D Test
    // Create a 2x3x4 tensor
    constexpr int DEPTH = 2;
    float data3d[DEPTH * HEIGHT * WIDTH] = {
        // Layer 0
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        // Layer 1
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23
    };
    
    // Create a 3D tensor
    auto tensor3d = Tensor<float, 
        DimInfo<I_Dim, DEPTH, HEIGHT * WIDTH>, 
        DimInfo<J_Dim, HEIGHT, WIDTH>, 
        DimInfo<K_Dim, WIDTH, 1>
    >(data3d);
    
    // Create a matching Index type
    using Idx3D = Index<
        DimInfo<I_Dim, DEPTH, HEIGHT * WIDTH>, 
        DimInfo<J_Dim, HEIGHT, WIDTH>, 
        DimInfo<K_Dim, WIDTH, 1>
    >;
    
    // Test that tensor[idx] gives the same result as tensor[i][j][k]
    for (int i = 0; i < DEPTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int k = 0; k < WIDTH; k++) {
                // Create index from coordinates
                auto idx = Idx3D::from_coords(I_Dim(i), J_Dim(j), K_Dim(k));
                
                // Get data via traditional subscript
                float val1 = *tensor3d[I_Dim(i)][J_Dim(j)][K_Dim(k)];
                
                // Get data via index subscript
                float val2 = *tensor3d[idx];
                
                // Values should be identical
                EXPECT_EQ(val1, val2);
                EXPECT_EQ(val1, i * (HEIGHT * WIDTH) + j * WIDTH + k);
            }
        }
    }
}

// Test mixed subscription with both tensor and index
UTEST(TensorVariadic, MixedSubscription) {
    // Create a 3x4 tensor
    constexpr int HEIGHT = 3;
    constexpr int WIDTH = 4;
    float data[HEIGHT * WIDTH] = {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11
    };
    
    // Define dimension types
    class I_Dim : public Dim<I_Dim> { public: using Dim<I_Dim>::Dim; };
    class J_Dim : public Dim<J_Dim> { public: using Dim<J_Dim>::Dim; };
    
    // Create tensor
    auto tensor = Tensor<float, 
        DimInfo<I_Dim, HEIGHT, WIDTH>, 
        DimInfo<J_Dim, WIDTH, 1>
    >(data);
    
    // Create an index for just J dimension
    using JIdx = Index<DimInfo<J_Dim, WIDTH, 1>>;
    
    // Test mixed tensor[i][jIdx]
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            // Create J index
            auto jIdx = JIdx::from_coords(J_Dim(j));
            
            // Get data via mixed subscription
            float val1 = *tensor[I_Dim(i)][jIdx];
            
            // Compare with standard subscription
            float val2 = *tensor[I_Dim(i)][J_Dim(j)];
            
            EXPECT_EQ(val1, val2);
        }
    }
}
