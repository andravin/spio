#include "spio/semaphore.cuh"

extern "C"
{
    __global__ void warp_semaphore(long long *__restrict__ event_types, long long *__restrict__ event_times)
    {
        constexpr unsigned warps = 16;
        constexpr unsigned max_count = 8;
        constexpr unsigned iters_per_warp = 64;
        constexpr unsigned events_per_warp = iters_per_warp * 2;
        constexpr unsigned num_iters = warps * iters_per_warp;

        __shared__ unsigned next_reservation;
        __shared__ unsigned next_execution;

        spio::WarpSemaphore sem(
            &next_reservation, &next_execution, max_count, threadIdx.x);

        __syncthreads();

        auto warp_idx = threadIdx.x / 32;
        auto lane_idx = threadIdx.x % 32;

        event_types += warp_idx * events_per_warp;
        event_times += warp_idx * events_per_warp;

        for (int i = 0; i < iters_per_warp; ++i)
        {
            sem.acquire();

            if (lane_idx == 0)
            {
                event_types[i * 2 + 0] = 1;
                event_times[i * 2 + 0] = clock64();

                auto sleepy_time = ((i * 2 + threadIdx.x) % 100) + 50;
                __nanosleep(sleepy_time);

                event_types[i * 2 + 1] = 0;
                event_times[i * 2 + 1] = clock64();
            }
            __syncwarp();

            sem.release();
        }
    }
}