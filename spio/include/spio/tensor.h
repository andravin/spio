#ifndef SPIO_TENSOR_VARIADIC_H_
#define SPIO_TENSOR_VARIADIC_H_

#include "spio/macros.h"
#include "spio/dim.h"
#include "spio/dim_info.h"
#include "spio/compound_index.h"
#include "spio/allocator.h"
#include "spio/coordinates.h"

namespace spio {

    /// @brief Base class for tensor data.
    template <typename _data_type> class Data {
    public:
        using data_type = _data_type;
        static constexpr int element_size = sizeof(_data_type);

        DEVICE Data(_data_type* data = nullptr) : _data(data) {}

        DEVICE _data_type* get() const {
            return _data;
        }

        DEVICE void reset(_data_type* data) {
            _data = data;
        }

        DEVICE _data_type& operator*() const {
            return *_data;
        }

        DEVICE _data_type* operator->() const {
            return _data;
        }

    private:
        _data_type* _data;
    };

    // Forward declaration of Tensor class
    template <typename DataType, typename... DimInfos> class Tensor;

    // Implementation details
    namespace detail {

        /// @brief Update dimension info by replacing a given dimension with a new size.
        template <typename DimType, int SliceSize, typename Tuple, typename Result = tuple<>>
        struct update_dim_info;

        // Base case: empty input, return accumulated result
        template <typename DimType, int SliceSize, typename Result>
        struct update_dim_info<DimType, SliceSize, tuple<>, Result> {
            using type = Result;
        };

        // Recursive case
        template <typename DimType, int SliceSize, typename FirstInfo, typename... RestInfos,
                  typename... ResultTypes>
        struct update_dim_info<DimType, SliceSize, tuple<FirstInfo, RestInfos...>,
                               tuple<ResultTypes...>> {
            static constexpr bool is_match = is_same<DimType, typename FirstInfo::dim_type>::value;
            using updated_info = conditional_t<is_match,
                                               DimInfo<typename FirstInfo::dim_type, SliceSize,
                                                       FirstInfo::module_type::stride.get()>,
                                               FirstInfo>;
            using type = typename update_dim_info<DimType, SliceSize, tuple<RestInfos...>,
                                                  tuple<ResultTypes..., updated_info>>::type;
        };

        template <typename DimType, int SliceSize, typename... DimInfos>
        using update_dim_info_t =
            typename update_dim_info<DimType, SliceSize, tuple<DimInfos...>>::type;

        template <typename, typename> struct tensor_type_from_dim_info_tuple;

        template <typename DataType, typename... DimInfos>
        struct tensor_type_from_dim_info_tuple<DataType, tuple<DimInfos...>> {
            using tensor_type = Tensor<DataType, DimInfos...>;
        };

        // Helper to calculate maximum storage size needed with strides
        template <typename... DimInfos> struct calculate_storage_size;

        // Base case
        template <> struct calculate_storage_size<> {
            static constexpr int value = 1; // No dimensions, just one element
        };

        // Recursive case
        template <typename FirstDim, typename... RestDims>
        struct calculate_storage_size<FirstDim, RestDims...> {
            // Get size and stride for this dimension
            static constexpr int size = FirstDim::module_type::size.get();
            static constexpr int stride = FirstDim::module_type::stride.get();

            // Calculate max offset for this dimension plus rest of dims
            static constexpr int value =
                (size - 1) * stride + calculate_storage_size<RestDims...>::value;
        };

        template <typename CompoundIndex, typename Target, typename = void>
        struct has_apply_to : false_type {};

        template <typename CompoundIndex, typename Target>
        struct has_apply_to<CompoundIndex, Target,
                            void_t<decltype(declval<CompoundIndex>().apply_to(declval<Target&>()))>>
            : true_type {};

        template <typename CompoundIndex, typename Target>
        inline constexpr bool has_apply_to_v = has_apply_to<CompoundIndex, Target>::value;

        // Detect if a type is a Fold
        template <typename T> struct is_fold : false_type {};

        template <typename DimType, int Stride>
        struct is_fold<Fold<DimType, Stride>> : true_type {};

        template <typename T> inline constexpr bool is_fold_v = is_fold<T>::value;

        // Detect if a type is a Module
        template <typename T> struct is_module : false_type {};

        template <typename DimType, int Size, int Stride>
        struct is_module<Module<DimType, Size, Stride>> : true_type {};

        template <typename T> inline constexpr bool is_module_v = is_module<T>::value;

    } // namespace detail

    template <typename DataType, typename... DimInfos> class BaseCursor;

    /// @brief Cursor with folded dimensions.
    /// Cursor is a class that represents a position in a tensor. It provides a subscript
    /// operator to access elements at a specific index in a given dimension.
    /// @tparam DataType the data type of the tensor
    /// @tparam DimInfos the dimension infos
    template <typename DataType, typename... DimInfos> class Cursor : public Data<DataType> {
    public:
        using Base = Data<DataType>;
        using data_type = DataType;
        using base_cursor_type = BaseCursor<DataType, DimInfos...>;

        DEVICE constexpr Cursor(DataType* data = nullptr, int offset = 0)
            : Base(data),
              _offset(offset) {}

        DEVICE constexpr data_type* get() const {
            return Base::get() + _offset;
        }

        DEVICE constexpr base_cursor_type rebase() const {
            return base_cursor_type(get());
        }

        // --- dimension presence trait ---
        template <typename DimType>
        static constexpr bool has_dimension_v =
            dim_traits::has_dimension<DimType, DimInfos...>::value;

    private:
        // Recursive helper for walking down Fold stride.
        template <class FoldType, bool NoError>
        DEVICE constexpr Cursor apply_fold_or_base(FoldType d) const {
            using DimType = typename FoldType::dim_type;
            constexpr int S = FoldType::stride.get();

            if constexpr (S > 1) {
                static_assert(S % 2 == 0,
                              "Fold stride must be divisible by 2 for this halving scheme.");
                constexpr int HalfStride = S / 2;
                using HalfFold = Fold<DimType, HalfStride>;

                if constexpr (has_dimension_v<HalfFold>) {
                    auto smaller = d.template fold<HalfStride>();
                    return (*this)[smaller];
                } else {
                    auto smaller = d.template fold<HalfStride>();
                    return apply_fold_or_base<HalfFold, NoError>(smaller);
                }
            } else {
                // S == 1: try base DimType
                if constexpr (has_dimension_v<DimType>) {
                    auto base_dim = d.unfold();
                    return (*this)[base_dim];
                } else {
                    // No matching fold or base dim found
                    if constexpr (NoError) {
                        // silent fallback
                        return *this;
                    } else {
                        // Let operator[] produce its compile-time error
                        static_assert(has_dimension_v<DimType>,
                                      "No matching dimension for Fold in this Cursor's DimInfos");
                        return *this; // unreachable, but required syntactically
                    }
                }
            }
        }

    public:
        template <typename T> DEVICE constexpr Cursor operator[](T t) const {
            if constexpr (detail::has_apply_to_v<T, Cursor>) {
                return t.apply_to(*this);
            } else if constexpr (has_dimension_v<T>) {
                constexpr int stride = dim_traits::dimension_stride<T, DimInfos...>::value.get();
                return Cursor(Base::get(), _offset + t.get() * stride);
            } else if constexpr (detail::is_fold_v<T>) {
                return apply_fold_or_base<T, false>(t);
            } else if constexpr (detail::is_module_v<T>) {
                return (*this)[t.to_fold()];
            } else {
                static_assert(has_dimension_v<T>,
                              "CompoundIndex type must be Dim-like or implement apply_to()");
                return *this;
            }
        }

        /// @brief  Apply the index in a specific dimension type if it exists.
        /// Also tries Fold resolution, but never errors if there is no match.
        template <typename DimType> DEVICE constexpr Cursor apply_index_if_found(DimType d) const {
            if constexpr (has_dimension_v<DimType>) {
                return (*this)[d];
            } else if constexpr (detail::is_fold_v<DimType> || detail::is_module_v<DimType>) {
                // Reuse fold logic, but in "no-error" mode.
                return apply_fold_or_base<DimType, /*NoError=*/true>(d);
            } else {
                return *this;
            }
        }

        /// @brief  Increment this cursor in a specific dimension type.
        /// @tparam Dim the dimension type in which the increment is applied.
        /// @param d The amount to increment by.
        /// @return a reference to the updated cursor.
        template <typename DimType> DEVICE Cursor& step(DimType d = 1) {
            // Keep current offset and reset base pointer.
            // We do this because the offset is const but the pointer is not.
            constexpr int stride =
                dim_traits::find_dim_info<DimType, DimInfos...>::info::module_type::stride.get();
            Base::reset(Base::get() + d.get() * stride);
            return *this;
        }

        DEVICE constexpr data_type& operator*() const {
            return *this->get();
        }

        DEVICE constexpr data_type* operator->() const {
            return this->get();
        }

    private:
        const int _offset;
    };

    /// @brief A class that implements a cursor with no offset.
    /// @tparam DataType the data type of the tensor
    /// @tparam DimInfos the dimension infos
    template <typename DataType, typename... DimInfos> class BaseCursor : public Data<DataType> {
    public:
        using Base = Data<DataType>;
        using data_type = DataType;
        using Base::Base;
        using Base::get;
        using cursor_type = Cursor<DataType, DimInfos...>;
        using base_cursor_type = BaseCursor<DataType, DimInfos...>;

        template <typename DimType>
        static constexpr bool has_dimension_v =
            dim_traits::has_dimension<DimType, DimInfos...>::value;

        /// @brief Subscript operator that returns a new Cursor at the specified dimension index.
        /// @tparam DimType the dimension to apply the subscript index to.
        /// @param d the subscript index.
        /// @return a new Cursor that points to the element at the specified dimension index.
        template <typename DimType> DEVICE constexpr cursor_type operator[](DimType d) const {
            return cursor_type(Base::get())[d];
        }

        template <typename DimType>
        DEVICE constexpr cursor_type apply_index_if_found(DimType d) const {
            return cursor_type(Base::get()).apply_index_if_found(d);
        }

        /// @brief  Increment the cursor in a specific dimension type.
        /// @tparam Dim the dimension type in which the increment is applied.
        /// @param d The amount to increment by.
        /// @return a reference to the updated cursor.
        template <typename DimType> DEVICE BaseCursor& step(DimType d = 1) {
            // Keep current offset and reset base pointer.
            // We do this because the offset is const but the pointer is not.
            constexpr int stride =
                dim_traits::find_dim_info<DimType, DimInfos...>::info::module_type::stride.get();
            Base::reset(Base::get() + d.get() * stride);
            return *this;
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
    template <typename DataType, typename... DimInfos> class Tensor : public Data<DataType> {
    public:
        using data_type = DataType;
        using Data<data_type>::Data;
        using Data<data_type>::get;

        using cursor_type = Cursor<DataType, DimInfos...>;
        using base_cursor_type = BaseCursor<DataType, DimInfos...>;

        // CompoundIndex type that uses tensor's size and strides.
        using index_type = CompoundIndex<DimInfos...>;

        // Total number of elements (product of all dimension sizes)
        // NOTE: this changes the meaning of the "size" method from
        // the previous implementation. Now it is just the number of elements.
        // It is not longer the storage size of the tensor. We need a
        // separate method to get the storage size.
        static constexpr int total_size = detail::product_sizes<DimInfos...>::value;

        // Helper variable template for cleaner usage
        template <typename DimType>
        static constexpr bool has_dimension_v =
            dim_traits::has_dimension<DimType, DimInfos...>::value;

        // Allocate a tensor on the stack.
        // The user would often initialize the StackAllocator object
        // with a pointer to shared memory, so that a smem buffer
        // is used as a stack for allocations and deallocations.
        DEVICE static Tensor allocate(StackAllocator& allocator) {
            return Tensor(allocator.allocate<data_type>(storage_size()));
        }

        // Deallocate a tensor from the stack.
        DEVICE void deallocate(StackAllocator& allocator) {
            allocator.deallocate<data_type>(get(), storage_size());
        }

        // For compatibility with existing code
        DEVICE static constexpr int size() {
            return total_size;
        }

        // Calculate actual storage size (accounting for strides)
        DEVICE static constexpr int storage_size() {
            return detail::calculate_storage_size<DimInfos...>::value;
        }

        // Return actual bytes needed, accounting for strides
        DEVICE static constexpr int num_bytes() {
            return storage_size() * sizeof(data_type);
        }

        // Get size for a specific dimension
        template <typename DimType> DEVICE static constexpr DimType size() {
            return dim_traits::dimension_size<DimType, DimInfos...>::value;
        }

        // Get all sizes as a Coordinates struct.
        DEVICE static constexpr auto sizes() {
            return Coordinates<typename DimInfos::dim_type...>(
                typename DimInfos::dim_type(DimInfos::module_type::size.get())...);
        }

        /// @brief Subscript operator with any dimension type.
        /// @tparam DimType the dimension to apply the subscript index to.
        /// @return a Cursor that points to the element at the specified position.
        template <typename DimType> DEVICE constexpr cursor_type operator[](DimType d) const {
            return cursor_type(get())[d];
        }

        template <typename DimType>
        DEVICE constexpr cursor_type apply_index_if_found(DimType d) const {
            return cursor_type(get()).apply_index_if_found(d);
        }

        /// @brief Get a cursor at a specific offset.
        /// @param offset the offset to get the cursor at.
        /// @return a cursor at the specified offset.
        DEVICE constexpr cursor_type offset(int offset) const {
            return cursor_type(get(), offset);
        }

        /// @brief Slice method to create a view with a different offset and size in one dimension.
        /// @tparam SliceSize the new size of the dimension
        /// @tparam SliceDimType the dimension to slice. SliceDimType is inferred from the type of
        /// the slice_start argument.
        /// @param slice_start the start index of the slice.
        /// @return a new Tensor that is a view of the original tensor with the specified
        /// dimension's size updated.
        template <int SliceSize, typename SliceDimType>
        DEVICE constexpr auto slice(SliceDimType slice_start) {
            using updated_infos = detail::update_dim_info_t<SliceDimType, SliceSize, DimInfos...>;
            using tensor_type =
                typename detail::tensor_type_from_dim_info_tuple<DataType,
                                                                 updated_infos>::tensor_type;
            return tensor_type((*this)[slice_start].get());
        }

        /// @brief Load data from a source cursor that points to a shared memory buffer.
        /// @tparam SrcCursorType the type of the source cursor.
        /// @param src the source cursor.
        template <typename SrcCursorType> DEVICE void load(SrcCursorType src) {
            load_impl<decltype(*this), SrcCursorType, DimInfos...>(*this, src);
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
        template <typename F> DEVICE void apply(F func) {
            apply_impl<F, decltype(*this), DimInfos...>(*this, func);
        }

        /// @brief Fill the tensor with zeros.
        DEVICE void zero() {
            auto zero_func = [](auto obj) { obj->zero(); };
            apply(zero_func);
        }

        /// @brief Fill the tensor with a specified value.
        /// @tparam Vector The value type
        /// @param value The value to fill with
        template <typename Vector> DEVICE void fill(Vector value) {
            auto fill_func = [value](auto obj) { obj->fill(value); };
            apply(fill_func);
        }

        template <typename Vector> DEVICE void add(Vector value) {
            auto add_func = [value](auto obj) { obj->add(value); };
            apply(add_func);
        }

    private:
        /// @brief Base case for loading data from a source cursor.
        template <typename DstCursorType, typename SrcCursorType>
        DEVICE static void load_impl(DstCursorType dst, SrcCursorType src) {
            dst->load(src.get());
        }

        /// @brief Recursive case for loading data from a source cursor.
        template <typename DstCursorType, typename SrcCursorType, typename FirstDimInfo,
                  typename... RestDimInfos>
        DEVICE static void load_impl(DstCursorType dst, SrcCursorType src) {
            using FirstDimType = typename FirstDimInfo::dim_type;
            auto size = FirstDimType(FirstDimInfo::module_type::size.get());
            for (auto i : range(size)) {
                load_impl<decltype(dst[i]), decltype(src[i]), RestDimInfos...>(dst[i], src[i]);
            }
        }

        /// @brief Base case for applying a function to tensor elements.
        template <typename F, typename CursorType>
        DEVICE static void apply_impl(CursorType obj, F func) {
            func(obj);
        }

        /// @brief Recursive case for applying a function to tensor elements.
        template <typename F, typename CursorType, typename FirstDimInfo, typename... RestDimInfos>
        DEVICE static void apply_impl(CursorType obj, F func) {
            using FirstDimType = typename FirstDimInfo::dim_type;
            auto size = FirstDimType(FirstDimInfo::module_type::size.get());
            for (auto i : range(size)) {
                apply_impl<F, decltype(obj[i]), RestDimInfos...>(obj[i], func);
            }
        }
    };
}

#endif
