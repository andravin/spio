#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include "parameters.h"

using namespace spio;

namespace
{
    __device__ float add_halves(__half2 x)
    {
        float2 x_f2 = __half22float2(x);
        return x_f2.x + x_f2.y;
    }

    __device__ float halves_diff_sq(__half2 x, float mean)
    {
        float2 x_f2 = __half22float2(x);
        float diff_x = x_f2.x - mean;
        float diff_y = x_f2.y - mean;
        return diff_x * diff_x + diff_y * diff_y;
    }

    // @brief Normalize the half-precision input.
    // @param x Two half-precision input values.
    // @param mean The LayerNorm2d mean.
    // @param r The reciprocal of the standard deviation.
    // @param weight The per-channel weight.
    // @param bias The per-channel bias.
    __device__ __half2 halves_norm(__half2 x, float mean, float r, float2 weight, float2 bias)
    {
        float2 x_f2 = __half22float2(x);
        float diff_x = x_f2.x - mean;
        float diff_y = x_f2.y - mean;

        float a_x = r;
        float a_y = r;
        if constexpr (Params::has_weight)
        {
            a_x *= weight.x;
            a_y *= weight.y;
        }
        float norm_x = diff_x * a_x;
        float norm_y = diff_y * a_y;
        if constexpr (Params::has_bias)
        {
            norm_x += bias.x;
            norm_y += bias.y;
        }
        return __float22half2_rn(make_float2(norm_x, norm_y));
    }

    constexpr __device__ int divup(int n, int d)
    {
        return (n + d - 1) / d;
    }

    constexpr unsigned FULL_MASK = 0xffffffff;

    /// @bried Reduce a partial sum over threads and warps.
    __device__ float reduce(float sum, ThreadIdx tidx, float *smem_sum_buf)
    {
        // Sum over threads.
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        }

        // Sum over warps.
        if constexpr (Block::warps_c > 1)
        {
            auto smem_sum = SmemSum(smem_sum_buf)[tidx.get<X>()];
            int warp_c_idx = threadIdx.x % 32;
            if (warp_c_idx == 0)
            {
                *smem_sum[tidx.get<WARP_C>()] = sum;
            }
            __syncthreads();
            if (warp_c_idx < Block::warps_c)
            {
                sum = *smem_sum[WARP_C(warp_c_idx)];
            }
            else
            {
                sum = 0.0f;
            }
            for (int offset = divup(Block::warps_c, 2); offset > 0; offset >>= 1)
            {
                sum += __shfl_down_sync(FULL_MASK, sum, offset);
            }
        }
        sum = __shfl_sync(FULL_MASK, sum, 0);
        return sum;
    }
}

extern "C"
{
    __global__ void SPIO_LAYERNORM_2D_KERNEL(
        __half2 *__restrict__ output_ptr,
        const uint4 *__restrict__ input_ptr,
        const float2 *__restrict__ weight_ptr,
        const float2 *__restrict__ bias_ptr)
    {
        //
        // Define the shared memory buffers.
        //
        __shared__ uint4 smem_input_buf[SmemInputStore::storage_size()];
        __shared__ float smem_sum_buf[Block::warps_x * Block::warps_c];
        __shared__ float smem_diff_sum_buf[Block::warps_x * Block::warps_c];

        //
        // Define the tile mappings.
        //
        int bx = []()
        {
            if constexpr (Mode::reverse_x)
            {
                return Block::blocks - 1 - blockIdx.x;
            }
            else
            {
                return blockIdx.x;
            };
        }();
        auto block_x = BLOCK_X(bx);

        // input -> smem
        Input input(input_ptr);
        SmemInputStore smem_input_store(smem_input_buf);

        // smem -> registers
        ThreadIdx tidx(threadIdx.x);
        auto c2_warp_lane = tidx.get<WARP_C>().fold<2>() + tidx.get<C2>();
        auto smem_input_load = SmemInputLoad(reinterpret_cast<const __half2 *>(smem_input_buf))[tidx.get<X>()].rebase();

        // Load the weight and bias.
        float2 weight[Params::c2_per_thread];
        float2 bias[Params::c2_per_thread];
        if constexpr (Params::has_weight)
        {
            for (int c64 = 0; c64 < Params::c2_per_thread; ++c64)
            {
                auto c2 = c2_warp_lane + c64 * 32;
                if (c2 < Output::size<C2>())
                {
                    weight[c64] = weight_ptr[c2.get()];
                    if constexpr (Params::has_bias)
                    {
                        bias[c64] = bias_ptr[c2.get()];
                    }
                }
                else
                {
                    weight[c64] = make_float2(0.0f, 0.0f);
                    if constexpr (Params::has_bias)
                    {
                        bias[c64] = make_float2(0.0f, 0.0f);
                    }
                }
            }
        }

        // Iterate over the x-dimension of the input tensor.
        for (int iter = 0; iter < Params::warp_x + 1; ++iter)
        {
            if (iter < Params::warp_x)
            {
                // Load the next input tile to shared memory asynchronously.
                auto x_iter = block_x.unfold() + X(Block::warps_x * iter);
                auto input_iter = input[x_iter];
                for (int idx = threadIdx.x; idx < SmemInputStore::size() / 2; idx += Block::threads)
                {
                    SmemInputStore::index_type smem_store_idx(idx);
                    bool thread_inbounds = (x_iter + smem_store_idx.get<X>()) < Input::size<X>();
                    int zfill = thread_inbounds ? 0 : sizeof(Input::data_type);
                    __pipeline_memcpy_async(
                        smem_input_store[PING_PONG(iter % 2)].get() + idx,
                        input_iter.get() + idx,
                        sizeof(Input::data_type),
                        zfill);
                }
                __pipeline_commit();
            }

            if (iter > 0)
            {
                int compute_iter = iter - 1;
                //
                // Load the input tile to registers.
                //
                __half2 input_tile[Params::c2_per_thread];
                __pipeline_wait_prior(compute_iter < Params::warp_x - 1 ? 1 : 0);
                __syncthreads();
                for (int c64 = 0; c64 < Params::c2_per_thread; ++c64)
                {
                    auto c2 = C2(c2_warp_lane + c64 * 32);
                    if (c2 < Output::size<C2>())
                    {
                        input_tile[c64] = *smem_input_load[PING_PONG(compute_iter % 2)][c2];
                    }
                    else
                    {
                        input_tile[c64] = __float22half2_rn(make_float2(0.0f, 0.0f));
                    }
                }

                //
                // Compute the mean.
                //
                float mean;
                {
                    float sum = 0.0f;

                    // Sum over registers.
                    for (int c64 = 0; c64 < Params::c2_per_thread; ++c64)
                    {
                        sum += add_halves(input_tile[c64]);
                    }
                    sum = reduce(sum, tidx, smem_sum_buf);
                    mean = sum / Params::c;
                }

                //
                // Compute the variance.
                //
                float var;
                {
                    float sum_diff_sq = 0.0f;

                    // Sum over registers.
                    for (int c64 = 0; c64 < Params::c2_per_thread; ++c64)
                    {
                        auto c2 = c2_warp_lane + c64 * 32;
                        if (c2 < Output::size<C2>())
                        {
                            sum_diff_sq += halves_diff_sq(input_tile[c64], mean);
                        }
                    }
                    sum_diff_sq = reduce(sum_diff_sq, tidx, smem_diff_sum_buf);
                    var = sum_diff_sq / Params::c;
                }

                // Reciprocal of the stdandard deviation.
                float rstd = rsqrtf(var + Params::eps);

                //
                // Compute and store the LayerNorm output.
                //
                {
                    auto x_iter = block_x.unfold() + Block::warps_x * compute_iter;
                    auto c2_warp_lane = tidx.get<WARP_C>().fold<2>() + tidx.get<C2>();
                    auto x = x_iter + tidx.get<X>();

                    auto output = Output(output_ptr)[x].rebase();

                    for (int c64 = 0; c64 < Params::c2_per_thread; ++c64)
                    {
                        __half2 norm = halves_norm(input_tile[c64], mean, rstd, weight[c64], bias[c64]);
                        auto c2 = c2_warp_lane + c64 * 32;
                        if (c2 < Output::size<C2>() && x < Output::size<X>())
                        {
                            *output[c2] = norm;
                        }
                    }
                }
            }
        }
    }
}
