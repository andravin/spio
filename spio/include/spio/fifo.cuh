#ifndef SPIO_FIFO_CUH_
#define SPIO_FIFO_CUH_

#ifndef UINT_MAX
#define UINT_MAX 4294967295
#endif

#include "spio/allocator.h"

namespace spio
{
    /// @brief  A warp-safe circular FIFO.
    /// The user must ensure the size of the FIFO never exceeds its capacity.
    ///
    /// To manage access to a limited number of resources, you can call WarpFifo::make_resourece_queue()
    /// to construct a FIFO that is initialized with the resource identifiers [0, num_resources).
    /// Then you acquire a resource-id by calling pop() and release it by calling push().
    /// @tparam Capacity
    template <unsigned Capacity>
    class WarpFifo
    {
        static_assert(Capacity <= 32, "Capacity must be less than or equal to 32.");
        static constexpr unsigned all_lanes_mask = 0xFFFFFFFF;
        static constexpr unsigned patience_ns = 20;

    public:
        using data_type = unsigned;

        static constexpr data_type sentinel_value = UINT_MAX;

        static constexpr unsigned capacity = Capacity;

        static constexpr unsigned smem_size = capacity + 2;

        /// @brief  Constructor
        /// The user must initialize the FIFO array head and tail indexes to their initial values
        /// before calling the constructor. Empty slots must be initialized to SENTINEL_VALUE.
        /// @param fifo pointer to the shared memory array that holds the FIFO slots.
        /// @param head pointer to the shared memory value that holds the index of the next element to read.
        /// @param tail pointer to the shared memory value that holds the index of the next element to write.
        /// @param tid the thread's unique identifier among the threads that participate in the FIFO.
        __device__ WarpFifo(data_type *fifo, unsigned *head, unsigned *tail, int tid)
            : _fifo(fifo), _head(head), _tail(tail), _is_first_lane(tid % 32 == 0)
        {
        }

        /// @brief  Create a resource queue.
        /// This function initializes the FIFO array with "resource" identifiers [0, num_resources)
        /// and the remaining slots with SENTINEL_VALUE. The head and tail indexes are initialized
        /// to 0 and num_resources, respectively. The user must ensure that num_resources <= Capacity.
        /// @param fifo pointer to the shared memory array that holds the FIFO slots.
        /// @param head pointer to the shared memory value that holds the index of the next element to read.
        /// @param tail pointer to the shared memory value that holds the index of the next element to write.
        /// @param tid the thread's unique identifier among the threads that participate in the FIFO.
        /// @param num_resources the number of resources to initialize the FIFO with. Must be less than or equal to Capacity.
        /// @return WarpFifo initialized with the requested number of resource ids [0, num_resources).
        __device__ static WarpFifo make_resource_queue(data_type *fifo, unsigned *head, unsigned *tail, int tid, int num_resources)
        {
            if (tid < capacity)
            {
                fifo[tid] = (tid < num_resources) ? tid : sentinel_value;
            }
            if (tid == 0)
            {
                *head = 0;
                *tail = num_resources;
            }
            return WarpFifo(fifo, head, tail, tid);
        }

        /// @brief Create a resource queue.
        /// This is a simpler form of the make_resource_queue() function that uses a single shared memory buffer.
        /// The buffer must be large enough to hold the FIFO array, the head and tail indexes. The total size
        /// of the buffer must be WarpFifo::smem_size.
        /// @param smem_buffer a shared memory buffer of size WarpFifo::smem_size
        /// @param tid the thread's unique identifier among the threads that participate in the FIFO.
        /// @param num_resources the number of resources to initialize the FIFO with. Must be less than or equal to Capacity.
        /// @return WarpFifo initialized with the requested number of resource ids [0, num_resources).
        __device__ static WarpFifo make_resource_queue(data_type *smem_buffer, int tid, int num_resources)
        {
            return make_resource_queue(
                smem_buffer,
                smem_buffer + capacity,
                smem_buffer + capacity + 1,
                tid,
                num_resources);
        }

        /// @brief  Allocate a resource queue.
        /// @param allocator the allocator object to use for allocating the shared memory buffer.
        /// @param tid the thread's unique identifier among the threads that participate in the FIFO.
        /// @param num_resources the number of resources to initialize the FIFO with. Must be less than or equal to Capacity.
        /// @return WarpFifo initialized with the requested number of resource ids [0, num_resources).
        __device__ static WarpFifo allocate_resource_queue(StackAllocator &allocator, int tid, int num_resources)
        {
            auto smem_buffer = allocator.allocate<data_type>(smem_size);
            return make_resource_queue(smem_buffer, tid, num_resources);
        }

        /// @brief  Deallocate the shared memory buffer used by the FIFO.
        /// @param allocator the allocator object that was used to allocate the shared memory buffer.
        __device__ void deallocate(StackAllocator &allocator)
        {
            allocator.deallocate(_fifo, smem_size);
        }

        /// @brief Push a value into the FIFO.
        /// Only the first thread in each warp performs the push operation. All threads in the warp synchronize afterwards.
        /// The user must ensure the FIFO does not overflow.
        __device__ void push(data_type value)
        {
            if (_is_first_lane)
            {
                auto idx = atomicAdd(_tail, 1);
                _fifo[idx % capacity] = value;
                __threadfence_block();
            }
            __syncwarp();
        }

        /// @brief Pop the next element from the FIFO and return it to all threads in the warp.
        /// Each warp pops one element from the FIFO. The popped value is broadcast to all lanes.
        /// This method performs a busy-wait loop until the FIFO is not empty.
        /// @return The next element in the FIFO.
        __device__ data_type pop()
        {
            data_type value;
            if (_is_first_lane)
            {
                auto idx = atomicAdd(_head, 1);
                auto slot = &_fifo[idx % capacity];
                for (
                    value = atomicExch(slot, sentinel_value);
                    value == sentinel_value;
                    value = atomicExch(slot, sentinel_value))
                {
                    __nanosleep(patience_ns);
                }
            }
            return __shfl_sync(all_lanes_mask, value, 0);
        }

        __device__ WarpFifo(const WarpFifo &) = delete;
        __device__ WarpFifo &operator=(const WarpFifo &) = delete;

    private:
        data_type *_fifo;
        unsigned *_head;
        unsigned *_tail;
        bool _is_first_lane;
    };

    /// @brief A guard that pops a value from a WarpFifo and pushes it back when the guard goes out of scope.
    /// This protects against the user forgetting to push the value back into the FIFO after using it.
    /// @tparam Capacity
    template <unsigned Capacity>
    class WarpFifoGuard
    {
    public:
        using Fifo = WarpFifo<Capacity>;

        using data_type = typename Fifo::data_type;

        /// @brief  Construct the guard by popping a value from the WarpFifo.
        /// @param fifo the WarpFifo to pop the value from.
        __device__ WarpFifoGuard(Fifo &fifo) : _fifo(fifo)
        {
            _value = _fifo.pop();
        }

        /// @brief  Destructor that pushes the value back into the WarpFifo.
        __device__ ~WarpFifoGuard()
        {
            _fifo.push(_value);
        }

        /// @brief  Get the value that was popped from the WarpFifo.
        __device__ data_type value() const
        {
            return _value;
        }

        __device__ WarpFifoGuard &operator=(const WarpFifoGuard &) = delete;
        __device__ WarpFifoGuard(const WarpFifoGuard &) = delete;

    private:
        Fifo &_fifo;
        data_type _value;
    };
}

#endif