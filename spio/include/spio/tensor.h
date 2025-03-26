#ifndef SPIO_TENSOR_VARIADIC_H_
#define SPIO_TENSOR_VARIADIC_H_

#include "spio/macros.h"
#include "spio/dim.h"
#include "spio/dim_info.h"
#include "spio/index.h"
#include "spio/allocator.h"
namespace spio
{

    /// @brief Base class for tensor data.
    template <typename _data_type>
    class Data
    {
    public:
        using data_type = _data_type;
        static constexpr int element_size = sizeof(_data_type);

        DEVICE Data(_data_type *data = nullptr) : _data(data) {}

        DEVICE _data_type *get() const { return _data; }
        DEVICE void reset(_data_type *data) { _data = data; }
        DEVICE _data_type &operator*() const { return *_data; }
        DEVICE _data_type *operator->() const { return _data; }

    private:
        _data_type *_data;
    };

    // Forward declaration of Tensor class
    template <typename DataType, typename... DimInfos>
    class Tensor;

    // Implementation details
    namespace detail
    {
        // Helper function to concatenate tuples
        template <typename T1, typename T2>
        struct tuple_cat_impl;

        // Specialization for empty first tuple
        template <typename... Ts>
        struct tuple_cat_impl<spio::detail::tuple<>, spio::detail::tuple<Ts...>>
        {
            using type = spio::detail::tuple<Ts...>;
        };

        // Specialization for non-empty first tuple
        template <typename T, typename... Ts, typename... Us>
        struct tuple_cat_impl<spio::detail::tuple<T, Ts...>, spio::detail::tuple<Us...>>
        {
            using type = spio::detail::tuple<T, Us...>;
        };

        // Helper function to concatenate two tuples
        template <typename T1, typename T2>
        using tuple_cat_t = typename tuple_cat_impl<T1, T2>::type;

        /// @brief Update dimension info by replacing a given dimension with a new size.
        /// @tparam DimType the dimension type to update
        /// @tparam SliceSize the new size of the dimension
        /// @tparam DimInfos the dimension infos
        template <typename DimType, int SliceSize, typename... DimInfos>
        struct update_dim_info;

        template <typename DimType, int SliceSize, typename FirstInfo, typename... RestInfos>
        struct update_dim_info<DimType, SliceSize, FirstInfo, RestInfos...>
        {
            static constexpr bool is_match = detail::is_same<DimType, typename FirstInfo::dim_type>::value;
            using current = detail::conditional_t<
                is_match,
                DimInfo<typename FirstInfo::dim_type, SliceSize, FirstInfo::module_type::stride.get()>,
                FirstInfo>;
            using next = typename update_dim_info<DimType, SliceSize, RestInfos...>::dim_type;
            using dim_type = tuple_cat_t<
                spio::detail::tuple<current>,
                next>;
        };

        template <typename DimType, int SliceSize>
        struct update_dim_info<DimType, SliceSize>
        {
            using dim_type = spio::detail::tuple<>;
        };

        template <typename, typename>
        struct tensor_type_from_dim_info_tuple;

        /// @brief Create a tensor type from a tuple of dimension infos.
        /// @tparam DataType the data type of the tensor
        /// @tparam DimInfos the dimension infos
        template <typename DataType, typename... DimInfos>
        struct tensor_type_from_dim_info_tuple<DataType, spio::detail::tuple<DimInfos...>>
        {
            using tensor_type = Tensor<DataType, DimInfos...>;
        };

        // Helper to calculate maximum storage size needed with strides
        template <typename... DimInfos>
        struct calculate_storage_size;

        // Base case
        template <>
        struct calculate_storage_size<>
        {
            static constexpr int value = 1; // No dimensions, just one element
        };

        // Recursive case
        template <typename FirstDim, typename... RestDims>
        struct calculate_storage_size<FirstDim, RestDims...>
        {
            // Get size and stride for this dimension
            static constexpr int size = FirstDim::module_type::size.get();
            static constexpr int stride = FirstDim::module_type::stride.get();

            // Calculate max offset for this dimension plus rest of dims
            static constexpr int value =
                (size - 1) * stride + calculate_storage_size<RestDims...>::value;
        };
    }

    template <typename DataType, typename... DimInfos>
    class BaseCursor;

    /// @brief Cursor with folded dimensions.
    /// Cursor is a class that represents a position in a tensor. It provides a subscript
    /// operator to access elements at a specific index in a given dimension.
    /// @tparam DataType the data type of the tensor
    /// @tparam DimInfos the dimension infos
    template <typename DataType, typename... DimInfos>
    class Cursor : public Data<DataType>
    {
    public:
        using Base = Data<DataType>;
        using data_type = DataType;
        using base_cursor_type = BaseCursor<DataType, DimInfos...>;

        DEVICE constexpr Cursor(DataType *data = nullptr, int offset = 0)
            : Base(data), _offset(offset) {}

        DEVICE constexpr data_type *get() const { return Base::get() + _offset; }

        /// @brief Create a base cursor with the offset folded into the base pointer.
        /// @return a new BaseCursor object
        DEVICE constexpr base_cursor_type rebase() const { return base_cursor_type(get()); }

        template <typename DimType>
        struct has_dimension
        {
            static constexpr bool value = dim_traits::has_dimension<DimType, DimInfos...>::value;
        };

        // Helper variable template for cleaner usage
        template <typename DimType>
        static constexpr bool has_dimension_v = has_dimension<DimType>::value;

        /// @brief Subscript operator that returns a new Cursor at the specified dimension index.
        /// @tparam DimType the dimension to apply the subscript index to.
        /// @param d the subscript index.
        /// @return a new Cursor that points to the element at the specified dimension index.
        template <typename DimType>
        DEVICE constexpr Cursor operator[](DimType d) const
        {
            constexpr int stride = dim_traits::dimension_stride<DimType, DimInfos...>::value.get();
            return Cursor(Base::get(), _offset + d.get() * stride);
        }

        /// @brief  Increment this cursor in a specific dimension type.
        /// @tparam Dim the dimension type in which the increment is applied.
        /// @param d The amount to increment by.
        /// @return a reference to the updated cursor.
        template <typename DimType>
        DEVICE Cursor &step(DimType d = 1)
        {
            // Keep current offset and reset base pointer.
            // We do this because the offset is const but the pointer is not.
            constexpr int stride = dim_traits::find_dim_info<DimType, DimInfos...>::info::module_type::stride.get();
            Base::reset(Base::get() + d.get() * stride);
            return *this;
        }

        /// @brief Subscript operator that takes an Index object and applies all dimensions
        /// @tparam IndexDimInfos The dimension infos in the Index
        /// @param idx The index containing coordinates for multiple dimensions
        /// @return A cursor pointing to the element at the specified position
        template <typename... IndexDimInfos>
        DEVICE constexpr Cursor operator[](Index<IndexDimInfos...> idx) const
        {
            return idx.apply_to(*this);
        }

        DEVICE constexpr data_type &operator*() const { return *this->get(); }
        DEVICE constexpr data_type *operator->() const { return this->get(); }

    private:
        const int _offset;
    };

    /// @brief A class that implements a cursor with no offset.
    /// @tparam DataType the data type of the tensor
    /// @tparam DimInfos the dimension infos
    template <typename DataType, typename... DimInfos>
    class BaseCursor : public Data<DataType>
    {
    public:
        using Base = Data<DataType>;
        using data_type = DataType;
        using Base::Base;
        using Base::get;
        using cursor_type = Cursor<DataType, DimInfos...>;
        using base_cursor_type = BaseCursor<DataType, DimInfos...>;

        template <typename DimType>
        struct has_dimension
        {
            static constexpr bool value = dim_traits::has_dimension<DimType, DimInfos...>::value;
        };

        // Helper variable template for cleaner usage
        template <typename DimType>
        static constexpr bool has_dimension_v = has_dimension<DimType>::value;

        /// @brief Subscript operator that returns a new Cursor at the specified dimension index.
        /// @tparam DimType the dimension to apply the subscript index to.
        /// @param d the subscript index.
        /// @return a new Cursor that points to the element at the specified dimension index.
        template <typename DimType>
        DEVICE constexpr cursor_type operator[](DimType d) const
        {
            return cursor_type(Base::get())[d];
        }

        /// @brief  Increment the cursor in a specific dimension type.
        /// @tparam Dim the dimension type in which the increment is applied.
        /// @param d The amount to increment by.
        /// @return a reference to the updated cursor.
        template <typename DimType>
        DEVICE BaseCursor &step(DimType d = 1)
        {
            // Keep current offset and reset base pointer.
            // We do this because the offset is const but the pointer is not.
            constexpr int stride = dim_traits::find_dim_info<DimType, DimInfos...>::info::module_type::stride.get();
            Base::reset(Base::get() + d.get() * stride);
            return *this;
        }

        /// @brief Subscript operator that takes an Index object and applies all dimensions
        /// @tparam IndexDimInfos The dimension infos in the Index
        /// @param idx The index containing coordinates for multiple dimensions
        /// @return A cursor pointing to the element at the specified position
        template <typename... IndexDimInfos>
        DEVICE constexpr cursor_type operator[](Index<IndexDimInfos...> idx) const
        {
            return cursor_type(Base::get())[idx];
        }
    };

    /// @brief Tensor class.
    /// Tensor is a class that represents a multi-dimensional array. It provides
    /// a subscript operator to access elements at a specific position. It also
    /// provides methods to get the size of a specific dimension and to slice the
    /// tensor along a specific dimension.
    ///
    /// Tensor uses "typed dimensions" to provide compile-time checks for dimension
    /// sizes and strides. Each dimension is a unique subclass of Dim. The subscript
    /// and slice methods are overloaded to by dimension type, so any attempt to
    /// use it is not possible to accidentally use a dimension index with the wrong dimension.
    ///
    /// Dim encapsulates an integer index and Dim subclasses implement arithmetic and comparison
    /// operators, so it is possible to add dimensions and compare them. But any attempt to
    /// use add or compare different dimension types will result in a compile-time error.

    /// @tparam DataType the data type of the tensor
    /// @tparam DimInfos the dimension infos
    template <typename DataType, typename... DimInfos>
    class Tensor : public Data<DataType>
    {
    public:
        using data_type = DataType;
        using Data<data_type>::Data;
        using Data<data_type>::get;

        using cursor_type = Cursor<DataType, DimInfos...>;
        using base_cursor_type = BaseCursor<DataType, DimInfos...>;

        // Index type that uses tensor's size and strides.
        using index_type = spio::Index<DimInfos...>;

        // Total number of elements (product of all dimension sizes)
        // NOTE: this changes the meaning of the "size" method from
        // the previous implementation. Now it is just the number of elements.
        // It is not longer the storage size of the tensor. We need a
        // separate method to get the storage size.
        static constexpr int total_size = detail::product_sizes<DimInfos...>::value;

        // Helper to check if this tensor has a specific dimension type
        template <typename DimType>
        struct has_dimension
        {
            static constexpr bool value = dim_traits::has_dimension<DimType, DimInfos...>::value;
        };

        // Helper variable template for cleaner usage
        template <typename DimType>
        static constexpr bool has_dimension_v = has_dimension<DimType>::value;

        // Allocate a tensor on the stack.
        // The user would often initialize the StackAllocator object
        // with a pointer to shared memory, so that a smem buffer
        // is used as a stack for allocations and deallocations.
        DEVICE static Tensor allocate(StackAllocator &allocator)
        {
            return Tensor(allocator.allocate<data_type>(storage_size()));
        }

        // Deallocate a tensor from the stack.
        DEVICE void deallocate(StackAllocator &allocator)
        {
            allocator.deallocate<data_type>(get(), storage_size());
        }

        // For compatibility with existing code
        DEVICE static constexpr int size() { return total_size; }

        // Calculate actual storage size (accounting for strides)
        DEVICE static constexpr int storage_size()
        {
            return detail::calculate_storage_size<DimInfos...>::value;
        }

        // Return actual bytes needed, accounting for strides
        DEVICE static constexpr int num_bytes()
        {
            return storage_size() * sizeof(data_type);
        }

        // Get size for a specific dimension
        template <typename DimType>
        DEVICE static constexpr DimType size()
        {
            return dim_traits::dimension_size<DimType, DimInfos...>::value;
        }

        /// @brief Subscript operator with any dimension type.
        /// @tparam DimType the dimension to apply the subscript index to.
        /// @return a Cursor that points to the element at the specified position.
        template <typename DimType>
        DEVICE constexpr cursor_type operator[](DimType d) const
        {
            return cursor_type(get())[d];
        }

        /// @brief Get a cursor at a specific offset.
        /// @param offset the offset to get the cursor at.
        /// @return a cursor at the specified offset.
        DEVICE constexpr cursor_type offset(int offset) const
        {
            return cursor_type(get(), offset);
        }

        /// @brief Subscript operator that takes an Index object and applies all dimensions
        /// @tparam IndexDimInfos The dimension infos in the Index
        /// @param idx The index containing coordinates for multiple dimensions
        /// @return A cursor pointing to the element at the specified position
        template <typename... IndexDimInfos>
        DEVICE constexpr cursor_type operator[](Index<IndexDimInfos...> idx) const
        {
            return cursor_type(get())[idx];
        }

        /// @brief Slice method to create a view with a different offset and size in one dimension.
        /// @tparam SliceSize the new size of the dimension
        /// @tparam SliceDimType the dimension to slice. SliceDimType is inferred from the type of the slice_start argument.
        /// @param slice_start the start index of the slice.
        /// @return a new Tensor that is a view of the original tensor with the specified dimension's size updated.
        template <int SliceSize, typename SliceDimType>
        DEVICE constexpr auto slice(SliceDimType slice_start)
        {
            using updated_infos = typename detail::update_dim_info<SliceDimType, SliceSize, DimInfos...>::dim_type;
            using tensor_type = typename detail::tensor_type_from_dim_info_tuple<DataType, updated_infos>::tensor_type;
            return tensor_type((*this)[slice_start].get());
        }

        /// @brief Load data from a source cursor that points to a shared memory buffer.
        /// @tparam SrcCursorType the type of the source cursor.
        /// @param src the source cursor.
        template <typename SrcCursorType>
        DEVICE void load(SrcCursorType src)
        {
            load_impl<DimInfos...>(*this, src);
        }

        /// @brief Apply a custom function to each element of the tensor
        /// @tparam F The function type (typically a lambda)
        /// @param func Function that takes a cursor and performs operations on it
        /// @details This is a power-user method that allows for custom element-wise
        ///          operations beyond the standard operations provided by the class.
        ///          The function should accept a cursor parameter and operate on it.
        /// @example
        ///   // Scale all elements by 2 and add 1
        ///   tensor.apply([](auto elem) { 
        ///     // The cursor's data_type must implement saxpy.
        ///     elem->saxpy(2.0f, 1.0f);
        ///   });
        template <typename F>
        DEVICE void apply(F func)
        {
            apply_impl<F, DimInfos...>(*this, func);
        }

        /// @brief Fill the tensor with zeros.
        DEVICE void zero()
        {
            auto zero_func = [] (auto obj) { obj->zero(); };
            apply(zero_func);
        }

        /// @brief Fill the tensor with a specified value.
        /// @tparam Vector The value type
        /// @param value The value to fill with
        template <typename Vector>
        DEVICE void fill(Vector value)
        {
            auto fill_func = [value] (auto obj) { obj->fill(value); };
            apply(fill_func);
        }

        template <typename Vector>
        DEVICE void add(Vector value)
        {
            auto add_func = [value] (auto obj) { obj->add(value); };
            apply(add_func);
        }

    private:
        /// @brief Base case for loading data from a source cursor.
        template <typename DstCursorType, typename SrcCursorType>
        DEVICE static void load_impl(DstCursorType dst, SrcCursorType src)
        {
            dst->load(src.get());
        }

        /// @brief Recursive case for loading data from a source cursor.
        /// Recursively iterate over each dimension of the source cursor,
        /// applying the dimension indexes to the source and destination cursors,
        /// and load the tensor elements from the source to the destination.
        /// @tparam FirstDimInfo the first dimension info.
        /// @tparam RestDimInfos the rest of the dimension infos.
        /// @tparam DstCursorType the type of the destination cursor.
        /// @tparam SrcCursorType the type of the source cursor.
        /// @param dst the destination cursor.
        /// @param src the source cursor.
        template <typename FirstDimInfo, typename... RestDimInfos, typename DstCursorType, typename SrcCursorType>
        DEVICE static void load_impl(DstCursorType dst, SrcCursorType src)
        {
            using FirstDimType = typename FirstDimInfo::dim_type;
            auto size = FirstDimType(FirstDimInfo::module_type::size.get());
            for (auto i : range(size))
            {
                load_impl<RestDimInfos...>(dst[i], src[i]);
            }
        }

        /// @brief Base case for applying a function to tensor elements.
        /// @tparam F The function type
        /// @tparam CursorType The cursor type to operate on
        /// @param obj The cursor
        /// @param func The function to apply
        template <typename F, typename CursorType>
        DEVICE static void apply_impl(CursorType obj, F func)
        {
            func(obj);
        }

        /// @brief Recursive case for applying a function to tensor elements.
        /// @tparam F The function type
        /// @tparam FirstDimInfo The first dimension info
        /// @tparam RestDimInfos The remaining dimension infos
        /// @tparam CursorType The cursor type
        /// @param obj The cursor to operate on
        /// @param func The function to apply
        template <typename F, typename FirstDimInfo, typename... RestDimInfos, typename CursorType>
        DEVICE static void apply_impl(CursorType obj, F func)
        {
            using FirstDimType = typename FirstDimInfo::dim_type;
            auto size = FirstDimType(FirstDimInfo::module_type::size.get());
            for (auto i : range(size))
            {
                apply_impl<F, RestDimInfos...>(obj[i], func);
            }
        }
    };
}

#endif
