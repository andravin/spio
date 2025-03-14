#include "utest.h"
#include "spio/tensor.h"
#include "spio/dim.h"

UTEST(Tensor1D, indexing)
{
    class D0 : public spio::Dim
    {
    public:
        using spio::Dim::Dim;
    };

    using Tensor = spio::Tensor1D<float, D0, 8>;

    float data[Tensor::size];
    Tensor tensor(data);

    for (auto d0 : spio::range(tensor.size_0))
    {
        auto p1 = tensor[d0].get();
        auto p2 = &data[d0.get()];
        EXPECT_EQ(p1, p2);
    }
}

// UTEST(Tensor2D, indexing)
// {
//     constexpr int D0 = 8;
//     constexpr int D1 = 16;

//     using Tensor = spio::Tensor2D<float, D1>;

//     float data[D0 * D1];

//     for (int d0 = 0; d0 < D0; ++d0)
//     {
//         for (int d1 = 0; d1 < D1; ++d1)
//         {
//             auto p1 = Tensor(data)._d0(d0)._d1(d1).get();
//             auto p2 = data + d0 * D1 + d1;
//             EXPECT_EQ(p1, p2);
//         }
//     }
// }

// UTEST(Tensor3D, indexing)
// {
//     constexpr int D0 = 3;
//     constexpr int D1 = 7;
//     constexpr int D2 = 13;

//     using Tensor = spio::Tensor3D<float, D1, D2>;

//     float data[D0 * D1 * D2];

//     for (int d0 = 0; d0 < D0; ++d0)
//     {
//         for (int d1 = 0; d1 < D1; ++d1)
//         {
//             for (int d2 = 0; d2 < D2; ++d2)
//             {
//                 auto p1 = Tensor(data)._d0(d0)._d1(d1)._d2(d2).get();
//                 auto p2 = data + d0 * D1 * D2 + d1 * D2 + d2;
//                 EXPECT_EQ(p1, p2);
//             }
//         }
//     }
// }

// UTEST(Tensor4D, indexing)
// {
//     constexpr int D0 = 3;
//     constexpr int D1 = 7;
//     constexpr int D2 = 13;
//     constexpr int D3 = 17;

//     using Tensor = spio::Tensor4D<float, D1, D2, D3>;

//     float data[D0 * D1 * D2 * D3];

//     for (int d0 = 0; d0 < D0; ++d0)
//     {
//         for (int d1 = 0; d1 < D1; ++d1)
//         {
//             for (int d2 = 0; d2 < D2; ++d2)
//             {
//                 for (int d3 = 0; d3 < D3; ++d3)
//                 {
//                     auto p1 = Tensor(data)._d0(d0)._d1(d1)._d2(d2)._d3(d3).get();
//                     auto p2 = data + d0 * D1 * D2 * D3 + d1 * D2 * D3 + d2 * D3 + d3;
//                     EXPECT_EQ(p1, p2);
//                 }
//             }
//         }
//     }
// }

// UTEST(Tensor5D, indexing)
// {
//     constexpr int D0 = 3;
//     constexpr int D1 = 7;
//     constexpr int D2 = 13;
//     constexpr int D3 = 17;
//     constexpr int D4 = 19;

//     using Tensor = spio::Tensor5D<float, D1, D2, D3, D4>;

//     float data[D0 * D1 * D2 * D3 * D4];

//     for (int d0 = 0; d0 < D0; ++d0)
//     {
//         for (int d1 = 0; d1 < D1; ++d1)
//         {
//             for (int d2 = 0; d2 < D2; ++d2)
//             {
//                 for (int d3 = 0; d3 < D3; ++d3)
//                 {
//                     for (int d4 = 0; d4 < D4; ++d4)
//                     {
//                         auto p1 = Tensor(data)._d0(d0)._d1(d1)._d2(d2)._d3(d3)._d4(d4).get();
//                         auto p2 = data + d0 * D1 * D2 * D3 * D4 + d1 * D2 * D3 * D4 + d2 * D3 * D4 + d3 * D4 + d4;
//                         EXPECT_EQ(p1, p2);
//                     }
//                 }
//             }
//         }
//     }
// }

// UTEST(Tensor6D, indexing)
// {
//     constexpr int D0 = 3;
//     constexpr int D1 = 7;
//     constexpr int D2 = 13;
//     constexpr int D3 = 17;
//     constexpr int D4 = 19;
//     constexpr int D5 = 23;

//     using Tensor = spio::Tensor6D<float, D1, D2, D3, D4, D5>;

//     float data[D0 * D1 * D2 * D3 * D4 * D5];

//     for (int d0 = 0; d0 < D0; ++d0)
//     {
//         for (int d1 = 0; d1 < D1; ++d1)
//         {
//             for (int d2 = 0; d2 < D2; ++d2)
//             {
//                 for (int d3 = 0; d3 < D3; ++d3)
//                 {
//                     for (int d4 = 0; d4 < D4; ++d4)
//                     {
//                         for (int d5 = 0; d5 < D5; ++d5)
//                         {
//                             auto p1 = Tensor(data)._d0(d0)._d1(d1)._d2(d2)._d3(d3)._d4(d4)._d5(d5).get();
//                             auto p2 = data + d0 * D1 * D2 * D3 * D4 * D5 + d1 * D2 * D3 * D4 * D5 + d2 * D3 * D4 * D5 + d3 * D4 * D5 + d4 * D5 + d5;
//                             EXPECT_EQ(p1, p2);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// UTEST(Tensor7D, indexing)
// {
//     constexpr int D0 = 3;
//     constexpr int D1 = 7;
//     constexpr int D2 = 13;
//     constexpr int D3 = 17;
//     constexpr int D4 = 4;
//     constexpr int D5 = 2;
//     constexpr int D6 = 8;

//     using Tensor = spio::Tensor7D<float, D1, D2, D3, D4, D5, D6>;

//     float data[D0 * D1 * D2 * D3 * D4 * D5 * D6];

//     for (int d0 = 0; d0 < D0; ++d0)
//     {
//         for (int d1 = 0; d1 < D1; ++d1)
//         {
//             for (int d2 = 0; d2 < D2; ++d2)
//             {
//                 for (int d3 = 0; d3 < D3; ++d3)
//                 {
//                     for (int d4 = 0; d4 < D4; ++d4)
//                     {
//                         for (int d5 = 0; d5 < D5; ++d5)
//                         {
//                             for (int d6 = 0; d6 < D6; ++d6)
//                             {
//                                 auto p1 = Tensor(data)._d0(d0)._d1(d1)._d2(d2)._d3(d3)._d4(d4)._d5(d5)._d6(d6).get();
//                                 auto p2 = data + d0 * D1 * D2 * D3 * D4 * D5 * D6 + d1 * D2 * D3 * D4 * D5 * D6 + d2 * D3 * D4 * D5 * D6 + d3 * D4 * D5 * D6 + d4 * D5 * D6 + d5 * D6 + d6;
//                                 EXPECT_EQ(p1, p2);
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// UTEST(Tensor2D, custom_stride)
// {
//     constexpr int D0 = 8;
//     constexpr int D1 = 16;
//     constexpr int D1_stride = 1;
//     constexpr int D0_stride = D1 + 1;

//     using Tensor = spio::Tensor2D<float, D1, D1_stride, D0_stride>;

//     float data[D0_stride * D1_stride];

//     for (int d0 = 0; d0 < D0; ++d0)
//     {
//         for (int d1 = 0; d1 < D1; ++d1)
//         {
//             auto p1 = Tensor(data)._d0(d0)._d1(d1).get();
//             auto p2 = data + d0 * D0_stride + d1 * D1_stride;
//             EXPECT_EQ(p1, p2);
//         }
//     }
// }

// UTEST(Tensor3D, custom_stride)
// {
//     constexpr int D0 = 3;
//     constexpr int D1 = 7;
//     constexpr int D2 = 13;
//     constexpr int D2_stride = 1;
//     constexpr int D1_stride = D2 + 1;
//     constexpr int D0_stride = D1 * D1_stride + 2;

//     using Tensor = spio::Tensor3D<float, D1, D2, D2_stride, D1_stride, D0_stride>;

//     float data[D0_stride * D1_stride * D0_stride];

//     for (int d0 = 0; d0 < D0; ++d0)
//     {
//         for (int d1 = 0; d1 < D1; ++d1)
//         {
//             for (int d2 = 0; d2 < D2; ++d2)
//             {
//                 auto p1 = Tensor(data)._d0(d0)._d1(d1)._d2(d2).get();
//                 auto p2 = data + d0 * D0_stride + d1 * D1_stride + d2 * D2_stride;
//                 EXPECT_EQ(p1, p2);
//             }
//         }
//     }
// }

// UTEST(Tensor4D, custom_stride)
// {
//     constexpr int D0 = 3;
//     constexpr int D1 = 7;
//     constexpr int D2 = 8;
//     constexpr int D3 = 5;
//     constexpr int D3_stride = 1;
//     constexpr int D2_stride = D3 + 1;
//     constexpr int D1_stride = D2 * D2_stride + 2;
//     constexpr int D0_stride = D1 * D1_stride + 3;

//     using Tensor = spio::Tensor4D<float, D1, D2, D3, D3_stride, D2_stride, D1_stride, D0_stride>;

//     float data[D0_stride * D1_stride * D2_stride * D3_stride];

//     for (int d0 = 0; d0 < D0; ++d0)
//     {
//         for (int d1 = 0; d1 < D1; ++d1)
//         {
//             for (int d2 = 0; d2 < D2; ++d2)
//             {
//                 for (int d3 = 0; d3 < D3; ++d3)
//                 {
//                     auto p1 = Tensor(data)._d0(d0)._d1(d1)._d2(d2)._d3(d3).get();
//                     auto p2 = data + d0 * D0_stride + d1 * D1_stride + d2 * D2_stride + d3 * D3_stride;
//                     EXPECT_EQ(p1, p2);
//                 }
//             }
//         }
//     }
// }

// UTEST(Tensor5D, custom_stride)
// {
//     constexpr int D0 = 3;
//     constexpr int D1 = 2;
//     constexpr int D2 = 8;
//     constexpr int D3 = 5;
//     constexpr int D4 = 2;
//     constexpr int D4_stride = 1;
//     constexpr int D3_stride = D4 + 1;
//     constexpr int D2_stride = D3 * D3_stride + 2;
//     constexpr int D1_stride = D2 * D2_stride + 3;
//     constexpr int D0_stride = D1 * D1_stride + 4;

//     using Tensor = spio::Tensor5D<float, D1, D2, D3, D4, D4_stride, D3_stride, D2_stride, D1_stride, D0_stride>;

//     float data[D0_stride * D1_stride * D2_stride * D3_stride * D4_stride];

//     for (int d0 = 0; d0 < D0; ++d0)
//     {
//         for (int d1 = 0; d1 < D1; ++d1)
//         {
//             for (int d2 = 0; d2 < D2; ++d2)
//             {
//                 for (int d3 = 0; d3 < D3; ++d3)
//                 {
//                     for (int d4 = 0; d4 < D4; ++d4)
//                     {
//                         auto p1 = Tensor(data)._d0(d0)._d1(d1)._d2(d2)._d3(d3)._d4(d4).get();
//                         auto p2 = data + d0 * D0_stride + d1 * D1_stride + d2 * D2_stride + d3 * D3_stride + d4 * D4_stride;
//                         EXPECT_EQ(p1, p2);
//                     }
//                 }
//             }
//         }
//     }
// }
