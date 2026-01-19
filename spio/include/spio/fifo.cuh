#ifndef SPIO_FIFO_CUH_
#define SPIO_FIFO_CUH_

#ifndef UINT_MAX
#define UINT_MAX 4294967295
#endif

#include "spio/allocator.h"

namespace spio {
    /// A warp-safe circular FIFO.
    ///
    /// The user must ensure the size of the FIFO never exceeds its capacity.
    /// To manage access to a limited number of resources, call make_resource_queue()
    /// to construct a FIFO initialized with resource identifiers [0, num_resources).
    /// Acquire a resource-id by calling pop() and release it by calling push().
    ///
    /// Template parameters:
    ///   Capacity   Maximum number of elements the FIFO can hold (must be <= 32).
    template <unsigned Capacity> class WarpFifo {
        static_assert(Capacity <= 32, "Capacity must be less than or equal to 32.");
        static constexpr unsigned all_lanes_mask = 0xFFFFFFFF;
        static constexpr unsigned patience_ns = 20;

    public:
        using data_type = unsigned;

        static constexpr data_type sentinel_value = UINT_MAX;

        static constexpr unsigned capacity = Capacity;

        static constexpr unsigned smem_size = capacity + 2;

        /// Constructs a WarpFifo from pre-initialized shared memory.
        ///
        /// The user must initialize the FIFO array, head, and tail indexes before calling.
        /// Empty slots must be initialized to sentinel_value.
        ///
        /// Parameters:
        ///   fifo   Pointer to the shared memory array holding FIFO slots.
        ///   head   Pointer to the index of the next element to read.
        ///   tail   Pointer to the index of the next element to write.
        ///   tid    Thread's unique identifier among participating threads.
        __device__ WarpFifo(data_type* fifo, unsigned* head, unsigned* tail, int tid)
            : _fifo(fifo),
              _head(head),
              _tail(tail),
              _is_first_lane(tid % 32 == 0) {}

        /// Creates a resource queue initialized with resource identifiers.
        ///
        /// Initializes the FIFO array with identifiers [0, num_resources) and remaining
        /// slots with sentinel_value. Sets head to 0 and tail to num_resources.
        ///
        /// Parameters:
        ///   fifo           Pointer to the shared memory array holding FIFO slots.
        ///   head           Pointer to the index of the next element to read.
        ///   tail           Pointer to the index of the next element to write.
        ///   tid            Thread's unique identifier among participating threads.
        ///   num_resources  Number of resources (must be <= Capacity).
        ///
        /// Returns:
        ///   WarpFifo initialized with resource ids [0, num_resources).
        __device__ static WarpFifo make_resource_queue(data_type* fifo, unsigned* head,
                                                       unsigned* tail, int tid, int num_resources) {
            if (tid < capacity) { fifo[tid] = (tid < num_resources) ? tid : sentinel_value; }
            if (tid == 0) {
                *head = 0;
                *tail = num_resources;
            }
            return WarpFifo(fifo, head, tail, tid);
        }

        /// Creates a resource queue using a single shared memory buffer.
        ///
        /// Simplified form that uses one contiguous buffer for the FIFO array, head, and tail.
        ///
        /// Parameters:
        ///   smem_buffer    Shared memory buffer of size WarpFifo::smem_size.
        ///   tid            Thread's unique identifier among participating threads.
        ///   num_resources  Number of resources (must be <= Capacity).
        ///
        /// Returns:
        ///   WarpFifo initialized with resource ids [0, num_resources).
        __device__ static WarpFifo make_resource_queue(data_type* smem_buffer, int tid,
                                                       int num_resources) {
            return make_resource_queue(smem_buffer, smem_buffer + capacity,
                                       smem_buffer + capacity + 1, tid, num_resources);
        }

        /// Allocates and creates a resource queue using a StackAllocator.
        ///
        /// Parameters:
        ///   allocator      Allocator for the shared memory buffer.
        ///   tid            Thread's unique identifier among participating threads.
        ///   num_resources  Number of resources (must be <= Capacity).
        ///
        /// Returns:
        ///   WarpFifo initialized with resource ids [0, num_resources).
        __device__ static WarpFifo allocate_resource_queue(StackAllocator& allocator, int tid,
                                                           int num_resources) {
            auto smem_buffer = allocator.allocate<data_type>(smem_size);
            return make_resource_queue(smem_buffer, tid, num_resources);
        }

        /// Deallocates the shared memory buffer used by the FIFO.
        ///
        /// Parameters:
        ///   allocator   The allocator that was used to allocate the buffer.
        __device__ void deallocate(StackAllocator& allocator) {
            allocator.deallocate(_fifo, smem_size);
        }

        /// Pushes a value into the FIFO.
        ///
        /// Only the first thread in each warp performs the push. All threads synchronize after.
        /// The user must ensure the FIFO does not overflow.
        __device__ void push(data_type value) {
            if (_is_first_lane) {
                auto idx = atomicAdd(_tail, 1);
                _fifo[idx % capacity] = value;
                __threadfence_block();
            }
            __syncwarp();
        }

        /// Pops the next element from the FIFO and broadcasts it to all warp threads.
        ///
        /// Each warp pops one element. Performs a busy-wait loop until the FIFO is not empty.
        ///
        /// Returns:
        ///   The next element in the FIFO.
        __device__ data_type pop() {
            data_type value;
            if (_is_first_lane) {
                auto idx = atomicAdd(_head, 1);
                auto slot = &_fifo[idx % capacity];
                for (value = atomicExch(slot, sentinel_value); value == sentinel_value;
                     value = atomicExch(slot, sentinel_value)) {
                    __nanosleep(patience_ns);
                }
            }
            return __shfl_sync(all_lanes_mask, value, 0);
        }

        __device__ WarpFifo(const WarpFifo&) = delete;
        __device__ WarpFifo& operator=(const WarpFifo&) = delete;

    private:
        data_type* _fifo;
        unsigned* _head;
        unsigned* _tail;
        bool _is_first_lane;
    };

    /// RAII guard that pops a value from a WarpFifo and pushes it back on destruction.
    ///
    /// Protects against forgetting to push the value back after using it.
    ///
    /// Template parameters:
    ///   Capacity   Capacity of the underlying WarpFifo.
    template <unsigned Capacity> class WarpFifoGuard {
    public:
        using Fifo = WarpFifo<Capacity>;

        using data_type = typename Fifo::data_type;

        /// Constructs the guard by popping a value from the WarpFifo.
        ///
        /// Parameters:
        ///   fifo   The WarpFifo to pop the value from.
        __device__ WarpFifoGuard(Fifo& fifo) : _fifo(fifo) {
            _value = _fifo.pop();
        }

        /// Destructor that pushes the value back into the WarpFifo.
        __device__ ~WarpFifoGuard() {
            _fifo.push(_value);
        }

        /// Returns the value that was popped from the WarpFifo.
        __device__ data_type value() const {
            return _value;
        }

        __device__ WarpFifoGuard& operator=(const WarpFifoGuard&) = delete;
        __device__ WarpFifoGuard(const WarpFifoGuard&) = delete;

    private:
        Fifo& _fifo;
        data_type _value;
    };
}

#endif