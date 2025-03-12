#include "spio/semaphore.cuh"
#include "spio/fifo.cuh"

extern "C"
{
    __global__ void warp_fifo(long long *__restrict__ event_types, long long *__restrict__ event_times)
    {
        constexpr unsigned warps = 16;
        constexpr unsigned num_resources = 8;
        constexpr unsigned iters_per_warp = 64;
        constexpr unsigned events_per_warp = iters_per_warp * 2;

        __shared__ unsigned fifo_resources[num_resources];
        __shared__ unsigned fifo_head;
        __shared__ unsigned fifo_tail;
        __shared__ unsigned sema_next_reservation;
        __shared__ unsigned sema_next_execution;

        auto fifo = spio::WarpFifo<num_resources>::make_resource_queue(
            fifo_resources, &fifo_head, &fifo_tail, threadIdx.x, num_resources);

        auto sema = spio::WarpSemaphore(&sema_next_reservation, &sema_next_execution, num_resources, threadIdx.x);

        __syncthreads();

        auto warp_idx = threadIdx.x / 32;
        auto lane_idx = threadIdx.x % 32;

        event_types += warp_idx * events_per_warp;
        event_times += warp_idx * events_per_warp;

        for (int i = 0; i < iters_per_warp; ++i)
        {
            sema.acquire();

            auto resource_id = fifo.pop();

            if (lane_idx == 0)
            {
                event_types[i * 2 + 0] = resource_id;
                event_times[i * 2 + 0] = clock64();
                auto sleepy_time = ((i * 2 + threadIdx.x) % 100) + 50;
                __nanosleep(sleepy_time);
            }
            __syncwarp();

            fifo.push(resource_id);

            if (lane_idx == 0)
            {
                event_types[i * 2 + 1] = resource_id + num_resources;
                event_times[i * 2 + 1] = clock64();
            }

            sema.release();
        }
    }
}