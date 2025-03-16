#include <spio/index_variadic.h>
#include <spio/tensor_variadic.h>

using namespace spio;

extern "C"
{
    __global__ void index_variadic(float *outputs, float *inputs)
    {
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

        using Idx = Index<DimInfo<I_Dim, 64, 4>, DimInfo<J_Dim, 4, 1>>;
        using T = Tensor<float, DimInfo<J_Dim, 4, 64>, DimInfo<I_Dim, 64, 1>>;

        T output_tensor(outputs);
        T input_tensor(inputs);

        Idx idx(threadIdx.x);
        *output_tensor[idx] = *input_tensor[idx];
    }
}
