#ifndef SPIO_SEMAPHORE_CUH_
#define SPIO_SEMAPHORE_CUH_

#ifndef UINT_MAX
#define UINT_MAX 4294967295
#endif

namespace spio
{
    /// @brief A fair, warp-based semaphore.
    /// This semaphore ensures that threads are served in the order they requested access.
    class WarpSemaphore
    {
        static constexpr unsigned patience_ns = 20;

    public:
        using data_type = unsigned;

        /// @brief  Constructor
        /// You must synchronize the warps that participate in the semaphore after calling this constructor
        /// and before calling acquire() or release().
        /// @param next_reservation pointer to a shared memory variable that holds the next reservation index.
        /// @param next_execution pointer to a shared memory variable that holds the next execution index.
        /// @param count the initial semaphore count.
        /// @param tid this thread's unique identifier among the threads that participate in the semaphore.
        __device__ WarpSemaphore(
            data_type *__restrict__ next_reservation,
            data_type *__restrict__ next_execution,
            data_type count,
            unsigned tid)
            : _next_reservation(next_reservation),
              _next_execution(next_execution),
              _is_first_lane((tid % 32) == 0)
        {
            if (tid == 0)
            {
                *_next_execution = count;
                *_next_reservation = 0;
            }
            __syncwarp();
        }

        /// @brief  Acquire the semaphore for this warp.
        __device__ void acquire()
        {
            if (_is_first_lane)
            {
                auto seqno = atomicAdd(_next_reservation, 1);

                volatile data_type *next_execution = _next_execution;

                // Wait until *next_execution is greater than our sequence number.
                // This is a busy-wait loop, but it is only executed by the first thread in each warp.
                // The comparison is correct even if seqno overflows because unsigned integers wrap around.
                while (static_cast<int>(*next_execution - seqno) <= 0)
                {
                    __nanosleep(patience_ns);
                }
            }
            __syncwarp();
        }

        /// @brief Release this warp from the semaphore.
        __device__ void release()
        {
            __threadfence_block();
            __syncwarp();
            if (_is_first_lane)
            {
                atomicAdd(_next_execution, 1);
            }
            __syncwarp();
        }

        WarpSemaphore(const WarpSemaphore &) = delete;
        WarpSemaphore &operator=(const WarpSemaphore &) = delete;

    private:
        data_type *_next_reservation;
        data_type *_next_execution;
        bool _is_first_lane;
    };
}

#endif