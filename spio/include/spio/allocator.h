#ifndef SPIO_ALLOCATOR_H_
#define SPIO_ALLOCATOR_H_

#include "spio/macros.h"

namespace spio
{
    /// @brief Stack based allocator for shared memory.
    ///
    /// This is a simple allocator that uses a pointer to shared memory and
    /// increments the pointer as memory is allocated. It does not check for
    /// memory leaks or double deallocation. It is the user's responsibility to
    /// ensure that the memory is deallocated in the reverse order of allocation.
    ///
    /// It is not necessary to deallocate shared memory at the end of the kernel,
    /// as it is automatically freed when the kernel exits.
    ///
    /// The purpose of the deallocate function is to allow the user to free memory
    /// so that it can be reused for another allocation. This is useful for shared
    /// memory buffers that are only used in one stage of the kernel, such as input loading,
    /// main loop, and output writing. The user can deallocate the memory that is specific
    /// to a stage after it is no longer needed, allowing the memory to be reused for the next stage.
    class StackAllocator
    {
        /// @brief Calculate the size of an array of type T in terms of unsigned integers.
        /// @param size The number of elements in the array.
        /// @return The size of the array in terms of unsigned integers.
        /// @tparam T The type of the elements in the array.
        template <typename T>
        static constexpr DEVICE int _unsigned_size(int size) { return size * (sizeof(T) / sizeof(unsigned)); }

        /// @brief  Cast the shared memory pointer to a pointer of type T.
        template <typename T>
        DEVICE T *_cast() { return reinterpret_cast<T *>(_stack_ptr); }

        /// @brief Compile time check to ensure that the size of T is a multiple of the size of unsigned and greater than zero.
        template <typename T>
        DEVICE void _static_check()
        {
            static_assert(sizeof(T) % sizeof(unsigned) == 0, "Size of T must be a multiple of the size of unsigned.");
            static_assert(sizeof(T) > 0, "Size of T must be greater than zero.");
        }

    public:
        /// @brief Construct the allocator with a pointer to shared memory.
        /// @details The pointer must be aligned to the size of unsigned.
        DEVICE StackAllocator(void *smem_ptr) : _stack_ptr(reinterpret_cast<unsigned *>(smem_ptr)) {}

        /// @brief Allocate an array of type T in shared memory.
        /// @tparam T the data-type of the array elements.
        /// @param size the number of elements in the array.
        /// @return a pointer to the array.
        template <typename T>
        DEVICE T *allocate(int size)
        {
            _static_check<T>();
            auto ptr = _cast<T>();
            _stack_ptr += _unsigned_size<T>(size);
            return ptr;
        }

        /// @brief Deallocate an array of type T in shared memory.
        /// @details The user must deallocate the memory in the reverse order of allocation.
        /// @details The pointer to the array is only used to infer the data-type.
        /// @tparam T the data-type of the array elements.
        /// @param ptr the pointer to the array to deallocate.
        /// @param size the number of elements in the array.
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
