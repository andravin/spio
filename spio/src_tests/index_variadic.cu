#include <spio/index_variadic.h>
#include <spio/tensor_variadic.h>

using namespace spio;

extern "C"
{
    __global__ void index_variadic(float *outputs, float *inputs)
    {
        class I : public Dim<I>
        {
        public:
            using Dim<I>::Dim;
        };

        class J : public Dim<J>
        {
        public:
            using Dim<J>::Dim;
        };

        using Idx = Index<DimInfo<I, 64, 4>, DimInfo<J, 4, 1>>;
        using T = Tensor<float, DimInfo<J, 4, 64>, DimInfo<I, 64, 1>>;

        T output_tensor(outputs);
        T input_tensor(inputs);

        Idx idx(threadIdx.x);
        *output_tensor[idx] = *input_tensor[idx];
    }
}
