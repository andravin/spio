#ifndef SPIO_ALLOCATOR_H_
#define SPIO_ALLOCATOR_H_

#include "spio/macros.h"

namespace spio
{
    /// Stack-based allocator for shared memory.
    ///
    /// Uses a pointer to shared memory and increments it as memory is allocated.
    /// Does not check for memory leaks or double deallocation. The user must
    /// deallocate memory in the reverse order of allocation.
    ///
    /// Shared memory is automatically freed when the kernel exits, so deallocation
    /// is only needed to reuse memory for another allocation within the kernel.
    class StackAllocator
    {
        /// Calculates the size of an array of type T in unsigned integer units.
        template <typename T>
        static constexpr DEVICE int _unsigned_size(int size) { return size * (sizeof(T) / sizeof(unsigned)); }

        /// Casts the shared memory pointer to a pointer of type T.
        template <typename T>
        DEVICE T *_cast() { return reinterpret_cast<T *>(_stack_ptr); }

        /// Compile-time check that sizeof(T) is a positive multiple of sizeof(unsigned).
        template <typename T>
        DEVICE void _static_check()
        {
            static_assert(sizeof(T) % sizeof(unsigned) == 0, "Size of T must be a multiple of the size of unsigned.");
            static_assert(sizeof(T) > 0, "Size of T must be greater than zero.");
        }

    public:
        /// Constructs the allocator with a pointer to shared memory.
        ///
        /// The pointer must be aligned to the size of unsigned.
        DEVICE StackAllocator(void *smem_ptr) : _stack_ptr(reinterpret_cast<unsigned *>(smem_ptr)) {}

        /// Allocates an array of type T in shared memory.
        ///
        /// Template parameters:
        ///   T       Element type.
        ///
        /// Parameters:
        ///   size    Number of elements to allocate.
        ///
        /// Returns:
        ///   Pointer to the allocated array.
        template <typename T>
        DEVICE T *allocate(int size)
        {
            _static_check<T>();
            auto ptr = _cast<T>();
            _stack_ptr += _unsigned_size<T>(size);
            return ptr;
        }

        /// Deallocates an array of type T in shared memory.
        ///
        /// Must deallocate in reverse order of allocation. The pointer is only
        /// used to infer the element type.
        ///
        /// Template parameters:
        ///   T       Element type.
        ///
        /// Parameters:
        ///   ptr     Pointer to the array (used only for type inference).
        ///   size    Number of elements to deallocate.
        template <typename T>
        DEVICE void deallocate(T *ptr, int size)
        {
            _static_check<T>();
            _stack_ptr -= _unsigned_size<T>(size);
        }

    private:
        unsigned *_stack_ptr;
    };
}

#endif
