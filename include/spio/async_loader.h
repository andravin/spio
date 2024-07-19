#ifndef SPIO_ASYNC_LOADER_H_
#define SPIO_ASYNC_LOADER_H_

namespace spio
{
__device__ constexpr int cdiv(int n, int d)
{
    return (n + d - 1) / d;
}

template <
    class Input_, class SmemInput_,
    int CHUNK_H_, int CHUNK_W_, int CHUNK_C8_,
    int THREADS_>
class AsyncLoader
{
public:
    static constexpr int N_ = Input_::N;
    static constexpr int H_ = Input_::Y;
    static constexpr int W_ = Input_::X;
    static constexpr int C8_ = Input_::C8;
    static constexpr int UNIT = 2;
    static constexpr int LOAD_VECTOR_SIZE = 16;
    static constexpr int MAX_BYTES_PER_BLOCK_LOAD = THREADS_ * LOAD_VECTOR_SIZE;
    static constexpr int NUM_BYTES_PER_ROW = CHUNK_W_ * CHUNK_C8_ * (8 * UNIT);
    static constexpr int NUM_BYTES_PER_CHUNK = CHUNK_H_ * NUM_BYTES_PER_ROW;
    static constexpr int NUM_LOADS_PER_CHUNK = cdiv(NUM_BYTES_PER_CHUNK, MAX_BYTES_PER_BLOCK_LOAD);
    static constexpr int NUM_THREAD_LOADS_PER_CHUNK = cdiv(NUM_BYTES_PER_CHUNK, LOAD_VECTOR_SIZE);

    __device__ AsyncLoader(uint4 *smem_in, const uint4 *in, int block_x, int block_c8)
    {
        for (int block_load_idx = 0; block_load_idx < NUM_LOADS_PER_CHUNK; ++block_load_idx)
        {
            int thread_load_idx = block_load_idx * THREADS_ + threadIdx.x;
            int thread_c8 = thread_load_idx % C8_;
            int thread_x = ((thread_load_idx / C8_) % CHUNK_W_);
            int thread_y = thread_load_idx / (CHUNK_W_ * C8_);

            int x = block_x + thread_x;
            int c8 = block_c8 + thread_c8;
            x_inbounds[block_load_idx] = x >= 0 && x < W_;
            do_load[block_load_idx] = thread_load_idx < NUM_THREAD_LOADS_PER_CHUNK;
            in_load[block_load_idx] = Input_(in).y(thread_y).x(x).c8(c8);
            in_smem_store[block_load_idx] = SmemInput_(smem_in).y(thread_y).x(thread_x).c8(thread_c8);
            thread_load_y[block_load_idx] = thread_y;
        }
    }

    __device__ void load(int block_y)
    {
        for (int block_load_idx = 0; block_load_idx < NUM_LOADS_PER_CHUNK; ++block_load_idx)
        {
            if (do_load[block_load_idx])
            {
                int y = thread_load_y[block_load_idx] + block_y;
                bool y_inbounds = (y >= 0 && y < H_);
                bool xy_inbounds = x_inbounds[block_load_idx] && y_inbounds;
                int zfill = xy_inbounds ? 0 : LOAD_VECTOR_SIZE;
                __pipeline_memcpy_async(
                    in_smem_store[block_load_idx].get(),
                    in_load[block_load_idx].y(block_y).get(),
                    LOAD_VECTOR_SIZE,
                    zfill);
            }
        }
    }

private:
    Input_ in_load[NUM_LOADS_PER_CHUNK];
    SmemInput_ in_smem_store[NUM_LOADS_PER_CHUNK];
    int thread_load_y[NUM_LOADS_PER_CHUNK];
    bool x_inbounds[NUM_LOADS_PER_CHUNK];
    bool do_load[NUM_LOADS_PER_CHUNK];
};
}
#endif