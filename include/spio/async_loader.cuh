#ifndef SPIO_ASYNC_LOADER_H_
#define SPIO_ASYNC_LOADER_H_

namespace spio
{
    /// @brief Return the ceiling of n / d.
    /// @param n an integer numerator.
    /// @param d an integer divisor.
    /// @return The ceiling of n / d as an integer.
    __device__ constexpr int cdiv(int n, int d)
    {
        return (n + d - 1) / d;
    }

    /// @brief  Return the maximum of two integers. Is a constexpr.
    __device__ constexpr int _max(int a, int b)
    {
        return a > b ? a : b;
    }

    template <class Input_, class SmemInput_, int FIRST_CHUNK_H_, int CHUNK_H_, int THREADS_>
    class AsyncLoader
    {
    public:
        static constexpr int N_ = Input_::N;
        static constexpr int H_ = Input_::Y;
        static constexpr int W_ = Input_::X;
        static constexpr int C8_ = Input_::C8;
        static constexpr int CHUNK_W_ = SmemInput_::X;
        static constexpr int CHUNK_C8_ = SmemInput_::C8;
        static constexpr int UNIT = 2;
        static constexpr int LOAD_VECTOR_SIZE = 16;
        static constexpr int MAX_BYTES_PER_BLOCK_LOAD = THREADS_ * LOAD_VECTOR_SIZE;
        static constexpr int NUM_BYTES_PER_ROW = CHUNK_W_ * CHUNK_C8_ * (8 * UNIT);

        static constexpr int NUM_BYTES_PER_CHUNK = CHUNK_H_ * NUM_BYTES_PER_ROW;
        static constexpr int NUM_LOADS_PER_CHUNK = cdiv(NUM_BYTES_PER_CHUNK, MAX_BYTES_PER_BLOCK_LOAD);
        static constexpr int NUM_THREAD_LOADS_PER_CHUNK = cdiv(NUM_BYTES_PER_CHUNK, LOAD_VECTOR_SIZE);

        static constexpr int NUM_BYTES_FIRST_CHUNK = FIRST_CHUNK_H_ * NUM_BYTES_PER_ROW;
        static constexpr int NUM_LOADS_FIRST_CHUNK = cdiv(NUM_BYTES_FIRST_CHUNK, MAX_BYTES_PER_BLOCK_LOAD);
        static constexpr int NUM_THREAD_LOADS_FIRST_CHUNK = cdiv(NUM_BYTES_FIRST_CHUNK, LOAD_VECTOR_SIZE);

        static constexpr int MAX_LOADS_PER_CHUNK = _max(NUM_LOADS_PER_CHUNK, NUM_LOADS_FIRST_CHUNK);

        static constexpr unsigned NUM_SMEM_ROWS = FIRST_CHUNK_H_ + CHUNK_H_;

        __device__ AsyncLoader(uint4 *smem_in, const uint4 *in, int block_n, int block_y, int block_x, int block_c8)
        {
            _block_y = block_y;
            for (int block_load_idx = 0; block_load_idx < MAX_LOADS_PER_CHUNK; ++block_load_idx)
            {
                int thread_load_idx = block_load_idx * THREADS_ + threadIdx.x;
                typename SmemInput_::Index idx(thread_load_idx);
                int thread_c8 = idx.c8();
                int thread_x = idx.x();
                int thread_y = idx.y();
                int y = block_y + thread_y;
                int x = block_x + thread_x;
                int c8 = block_c8 + thread_c8;
                x_c8_inbounds[block_load_idx] = x >= 0 && x < W_ && c8 < C8_;
                do_first_load[block_load_idx] = thread_load_idx < NUM_THREAD_LOADS_FIRST_CHUNK;
                do_load[block_load_idx] = thread_load_idx < NUM_THREAD_LOADS_PER_CHUNK;
                in_load[block_load_idx] = Input_(in).n(block_n).y(y).x(x).c8(c8);
                in_smem_store[block_load_idx] = SmemInput_(smem_in).x(thread_x).c8(thread_c8);
                thread_load_y[block_load_idx] = thread_y;
            }
        }

        __device__ void _do_load(int chunk_y, int block_load_idx)
        {
            unsigned local_y = thread_load_y[block_load_idx] + chunk_y;
            int global_y = static_cast<int>(local_y) + _block_y;
            bool y_inbounds = (global_y >= 0 && global_y < H_);
            bool xy_inbounds = x_c8_inbounds[block_load_idx] && y_inbounds;
            int zfill = xy_inbounds ? 0 : LOAD_VECTOR_SIZE;
            __pipeline_memcpy_async(
                in_smem_store[block_load_idx].y(local_y % NUM_SMEM_ROWS).get(),
                in_load[block_load_idx].y(chunk_y).get(),
                LOAD_VECTOR_SIZE,
                zfill);
        }

        __device__ void first_load()
        {
            for (int block_load_idx = 0; block_load_idx < NUM_LOADS_FIRST_CHUNK; ++block_load_idx)
            {
                if (do_first_load[block_load_idx])
                {
                    _do_load(0, block_load_idx);
                }
            }
            __pipeline_commit();
        }

        __device__ void load(int chunk_y)
        {
            for (int block_load_idx = 0; block_load_idx < NUM_LOADS_PER_CHUNK; ++block_load_idx)
            {
                if (do_load[block_load_idx])
                {
                    _do_load(chunk_y, block_load_idx);
                }
            }
            __pipeline_commit();
        }

    private:
        Input_ in_load[MAX_LOADS_PER_CHUNK];
        SmemInput_ in_smem_store[MAX_LOADS_PER_CHUNK];
        int thread_load_y[MAX_LOADS_PER_CHUNK];
        int _block_y;
        bool x_c8_inbounds[MAX_LOADS_PER_CHUNK];
        bool do_first_load[NUM_LOADS_FIRST_CHUNK];
        bool do_load[NUM_LOADS_PER_CHUNK];
    };
}
#endif