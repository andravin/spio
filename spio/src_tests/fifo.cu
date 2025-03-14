#include "spio/fifo.cuh"

extern "C"
{
    __global__ void warp_fifo(long long *__restrict__ event_types, long long *__restrict__ event_times)
    {
        constexpr unsigned num_resources = 8;
        constexpr unsigned iters_per_warp = 64;
        constexpr unsigned events_per_warp = iters_per_warp * 2;

        using Guard = spio::WarpFifoGuard<num_resources>;

        __shared__ unsigned smem[Guard::Fifo::smem_size];

        spio::StackAllocator allocator(smem);

        auto fifo = Guard::Fifo::allocate_resource_queue(allocator, threadIdx.x, num_resources);

        __syncthreads();

        auto warp_idx = threadIdx.x / 32;
        auto lane_idx = threadIdx.x % 32;

        event_types += warp_idx * events_per_warp;
        event_times += warp_idx * events_per_warp;

        for (int i = 0; i < iters_per_warp; ++i)
        {
            unsigned resource_id;
            {
                Guard guard(fifo);
                resource_id = guard.value();

                if (lane_idx == 0)
                {
                    event_types[i * 2 + 0] = resource_id;
                    event_times[i * 2 + 0] = clock64();
                    auto sleepy_time = ((i * 2 + threadIdx.x) % 100) + 50;
                    __nanosleep(sleepy_time);
                }
                __syncwarp();
            }

            if (lane_idx == 0)
            {
                event_types[i * 2 + 1] = resource_id + num_resources;
                event_times[i * 2 + 1] = clock64();
            }
        }
    }
}