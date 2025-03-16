#include "utest.h"
#include "spio/index_variadic.h"

using namespace spio;

class I_Dim : public Dim<I_Dim>
{
public:
    using Dim<I_Dim>::Dim;
};

class J_Dim : public Dim<J_Dim>
{
public:
    using Dim<J_Dim>::Dim;
};

class K_Dim : public Dim<K_Dim>
{
public:
    using Dim<K_Dim>::Dim;
};

template <>
struct utest_type_deducer<I_Dim>
{
    static void _(const I_Dim d)
    {
        UTEST_PRINTF("%d", d.get());
    }
};

template <>
struct utest_type_deducer<J_Dim>
{
    static void _(const J_Dim d)
    {
        UTEST_PRINTF("%d", d.get());
    }
};

template <>
struct utest_type_deducer<K_Dim>
{
    static void _(const K_Dim d)
    {
        UTEST_PRINTF("%d", d.get());
    }
};

UTEST(Index1D, get)
{
    using Idx = Index<DimInfo<I_Dim, 16, 1>>;
    for (unsigned offset = 0; offset < 16; ++offset)
    {
        Idx idx(offset);
        EXPECT_EQ(idx.get<I_Dim>(), offset);
    }
}

UTEST(Index2D, get)
{
    // I x J matrix with size 32 x 8.
    using Idx = Index<DimInfo<I_Dim, 32, 8>, DimInfo<J_Dim, 8, 1>>;
    for (unsigned offset = 0; offset < 256; ++offset)
    {
        Idx idx(offset);
        EXPECT_EQ(idx.get<I_Dim>(), offset / 8);
        EXPECT_EQ(idx.get<J_Dim>(), offset % 8);
    }
}

UTEST(Index1D, from_coords)
{
    using Idx = Index<DimInfo<I_Dim, 16, 1>>;
    
    // Test several coordinates
    for (int i = 0; i < 16; ++i)
    {
        // Create index from coordinate
        Idx idx = Idx::from_coords(I_Dim(i));
        
        // Verify the offset is correctly calculated
        EXPECT_EQ(idx.offset(), i);
        
        // Verify we can get back the same coordinate
        EXPECT_EQ(idx.get<I_Dim>(), i);
    }
}

UTEST(Index2D, from_coords)
{
    // I x J matrix with size 32 x 8.
    using Idx = Index<DimInfo<I_Dim, 32, 8>, DimInfo<J_Dim, 8, 1>>;
    
    // Test several coordinate combinations
    for (int i = 0; i < 32; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            // Create index from coordinates
            Idx idx = Idx::from_coords(I_Dim(i), J_Dim(j));
            
            // Verify the offset is correctly calculated
            EXPECT_EQ(idx.offset(), i * 8 + j);
            
            // Verify we can get back the same coordinates
            EXPECT_EQ(idx.get<I_Dim>(), i);
            EXPECT_EQ(idx.get<J_Dim>(), j);
        }
    }
}

UTEST(Index3D, from_coords)
{
    // I x J x K tensor with size 16 x 8 x 4
    // Stride for K is 1
    // Stride for J is 4 (= size of K)
    // Stride for I is 32 (= size of J * stride of J)
    using Idx = Index<
        DimInfo<I_Dim, 16, 32>, 
        DimInfo<J_Dim, 8, 4>, 
        DimInfo<K_Dim, 4, 1>
    >;
    
    // Test a few representative coordinates
    I_Dim i(3);
    J_Dim j(5);
    K_Dim k(2);
    
    // Calculate expected offset: i*32 + j*4 + k*1
    unsigned expected_offset = 3*32 + 5*4 + 2;
    
    // Create index from coordinates
    Idx idx = Idx::from_coords(i, j, k);
    
    // Verify offset
    EXPECT_EQ(idx.offset(), expected_offset);
    
    // Verify we get back the same coordinates
    EXPECT_EQ(idx.get<I_Dim>(), 3);
    EXPECT_EQ(idx.get<J_Dim>(), 5);
    EXPECT_EQ(idx.get<K_Dim>(), 2);
}

// Test round-trip conversions and out-of-bounds handling
UTEST(IndexAdvanced, round_trip)
{
    // I x J matrix with size 10 x 20
    using Idx = Index<DimInfo<I_Dim, 10, 20>, DimInfo<J_Dim, 20, 1>>;
    
    // 1. Normal case
    auto idx1 = Idx::from_coords(I_Dim(5), J_Dim(15));
    EXPECT_EQ(idx1.offset(), 5*20 + 15);
    EXPECT_EQ(idx1.get<I_Dim>(), 5);
    EXPECT_EQ(idx1.get<J_Dim>(), 15);
    
    // 2. Out-of-bounds coordinates wrap around due to modular arithmetic
    auto idx2 = Idx::from_coords(I_Dim(15), J_Dim(25)); // Out of bounds
    // Should be equivalent to I_Dim(5), J_Dim(5) due to modulo arithmetic
    EXPECT_EQ(idx2.get<I_Dim>(), 5);
    EXPECT_EQ(idx2.get<J_Dim>(), 5);
}

UTEST(IndexSize, total_size)
{
    // 1D index
    using Idx1D = Index<DimInfo<I_Dim, 16, 1>>;
    EXPECT_EQ(Idx1D::total_size, 16);
    EXPECT_EQ(Idx1D::size(), 16);
    
    // 2D index
    using Idx2D = Index<DimInfo<I_Dim, 32, 8>, DimInfo<J_Dim, 8, 1>>;
    EXPECT_EQ(Idx2D::total_size, 32 * 8);
    EXPECT_EQ(Idx2D::size(), 32 * 8);
    
    // 3D index
    using Idx3D = Index<
        DimInfo<I_Dim, 16, 32>, 
        DimInfo<J_Dim, 8, 4>, 
        DimInfo<K_Dim, 4, 1>
    >;
    EXPECT_EQ(Idx3D::total_size, 16 * 8 * 4);
    EXPECT_EQ(Idx3D::size(), 16 * 8 * 4);
}
