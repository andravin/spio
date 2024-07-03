extern "C"
{
// Pipeline Primitive Interface:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pipeline-interface
__global__ void conv_group_4_32w_4h_64c_test(
    uint4 *__restrict__ out,
    const uint4 *__restrict__ in,
    const uint4 *__restrict__ weights)
{
    constexpr int W = 32;
    constexpr int H = 4;
    constexpr int C = 32;
    constexpr int C8 = C / 8;
    constexpr int GROUP_WIDTH = 4;

    constexpr int BLOCK_W = W + 2;

    __shared__ uint4 smem_in[BLOCK_W * H * C8];
    __shared__ uint4 smem_weights[9 * C8 * GROUP_WIDTH];

    // XXX Use the shared memory to avoid compiler warnings.
    smem_in[0] = in[0];
    smem_weights[0] = smem_in[0];
    out[0] = smem_weights[0];

    // Load weights to shared memory.

    // Load weights to registers.

    // For each strip of input rows ..

    // .. load the strip of input rows to share memory.

    // .. for each horizontal shift of the input rows

    //     ..  for each input row in the strip

    //          .. load the shift of the input row to registers.

    //          .. multiply the row by the corresponding column of the weights tensor

    //          .. accumulate onto the appropriate outputs.

    //     .. store any completed outputs to global memory.
}
}
