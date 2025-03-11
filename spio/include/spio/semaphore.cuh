#ifndef SPIO_SEMAPHORE_CUH_
#define SPIO_SEMAPHORE_CUH_

#ifndef UINT_MAX
#define UINT_MAX 4294967295
#endif

namespace spio
{
    /// @brief A warp-based semaphore implementation that uses a circular buffer to manage waiters.
    /// This is a fair semaphore, meaning that it ensures that threads are served in the order they requested access.
    /// @tparam MAX_WAITERS
    template <unsigned MAX_WAITERS>
    class WarpSemaphore
    {
    public:
        using data_type = unsigned;

        static constexpr unsigned size = MAX_WAITERS;

        /// @brief  Constructor
        /// You must synchronize the warps that participate in the semaphore after calling this constructor
        /// and before calling acquire() or release().
        /// @param slots pointer to shared memory array of size MAX_WAITERS.
        /// @param next_reservation pointer to a shared memory variable that holds the next reservation index.
        /// @param next_execution pointer to a shared memory variable that holds the next execution index.
        /// @param count the initial semaphore count.
        /// @param tid this thread's unique identifier in the range [0, MAX_WAITERS * 32]
        __device__ WarpSemaphore(
            data_type *__restrict__ slots,
            data_type *__restrict__ next_reservation,
            data_type *__restrict__ next_execution,
            data_type count,
            unsigned tid)
            : _slots(slots),
              _next_reservation(next_reservation),
              _next_execution(next_execution),
              _first_lane((tid % 32) == 0)
        {
            if (tid < size)
            {
                // UINT_MAX is logically one less than 0,
                // so it is a sentinel value that indicates that the slot is empty.
                _slots[tid] = tid < count ? tid : UINT_MAX;
            }
            else if (tid == size)
            {
                *_next_execution = count;
            }
            else if (tid == size + 1)
            {
                *_next_reservation = 0;
            }
        }

        /// @brief  Acquire the semaphore for this warp.
        /// @return
        __device__ void acquire()
        {
            if (_first_lane)
            {
                auto seqno = atomicAdd(_next_reservation, 1);
                auto slot = &_slots[seqno % size];

                // Wait until the slot has our sequence number or a greater one.
                // This is a busy-wait loop, but it is only executed by the first thread in each warp.
                // The comparison works even if seqno overflows, because the slots are unsigned integers.
                while (static_cast<int>(*slot - seqno) < 0)
                    ;
            }
            __syncwarp();
        }

        /// @brief Release this warp from the semaphore.
        /// @return
        __device__ void release()
        {
            if (_first_lane)
            {
                auto seqno = atomicAdd(_next_execution, 1);
                _slots[seqno % size] = seqno;
            }
            __syncwarp();
        }

    private:
        volatile data_type *_slots;
        data_type *_next_reservation;
        data_type *_next_execution;
        bool _first_lane;
    };
}

#endif