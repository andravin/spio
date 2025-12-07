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

    // Forward declarations
    template <typename DataType, typename... DimInfos> class Tensor;
    template <typename DataType, typename... DimInfos> class Cursor;
    template <typename DataType, typename... DimInfos> class BaseCursor;

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

    } // namespace detail

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
        // Direct offset calculation for exact dimension match
        // Returns -1 if no exact match, otherwise returns the memory stride
        template <typename SourceDim> static constexpr int find_exact_match_stride() {
            return find_exact_match_stride_impl<SourceDim, DimInfos...>();
        }

        template <typename SourceDim> static constexpr int find_exact_match_stride_impl() {
            return -1; // No match
        }

        template <typename SourceDim, typename FirstInfo, typename... RestInfos>
        static constexpr int find_exact_match_stride_impl() {
            using TargetDimType = typename FirstInfo::dim_type;
            using SourceBaseDim = detail::get_base_dim_type_t<SourceDim>;
            using TargetBaseDim = detail::get_base_dim_type_t<TargetDimType>;
            constexpr int source_stride = detail::get_dim_stride<SourceDim>::value;
            constexpr int target_stride = detail::get_dim_stride<TargetDimType>::value;
            constexpr bool same_base = detail::is_same<SourceBaseDim, TargetBaseDim>::value;
            constexpr bool same_stride = (source_stride == target_stride);

            // Check for exact match: same base type and same stride
            if constexpr (same_base && same_stride) {
                // Found exact match - return the memory stride
                return FirstInfo::module_type::stride.get();
            } else {
                return find_exact_match_stride_impl<SourceDim, RestInfos...>();
            }
        }

        // Check if source dimension only matches exactly one target dimension
        template <typename SourceDim> static constexpr bool has_single_exact_match() {
            using matching = matching_dim_infos_t<SourceDim>;
            if constexpr (detail::tuple_size<matching>::value != 1) {
                return false;
            } else {
                return find_exact_match_stride<SourceDim>() >= 0;
            }
        }

        // Apply a single dimension to all matching DimInfos with optimization
        // Returns the offset contribution from this dimension
        template <typename SourceDim>
        DEVICE constexpr int apply_to_matching_impl(SourceDim, detail::tuple<>) const {
            return 0;
        }

        template <typename SourceDim, typename FirstInfo, typename... RestInfos>
        DEVICE constexpr int apply_to_matching_impl(SourceDim d,
                                                    detail::tuple<FirstInfo, RestInfos...>) const {
            using TargetDimType = typename FirstInfo::dim_type;
            constexpr int source_stride = detail::get_dim_stride<SourceDim>::value;
            constexpr int target_stride = detail::get_dim_stride<TargetDimType>::value;
            constexpr int target_size = FirstInfo::module_type::size.get();
            constexpr int target_max_value = (target_size - 1) * target_stride;
            constexpr int dim_stride = FirstInfo::module_type::stride.get();

            // Check if source has bounded size (is a Module)
            constexpr bool source_is_bounded = detail::is_bounded_v<SourceDim>;
            constexpr int source_size = detail::get_dim_size_v<SourceDim>;
            constexpr int source_max_value =
                source_is_bounded ? (source_size - 1) * source_stride : 0x7FFFFFFF;

            // Optimization 1: If source stride > target max value, source can't contribute
            if constexpr (source_stride > target_max_value) {
                return apply_to_matching_impl<SourceDim, RestInfos...>(
                    d, detail::tuple<RestInfos...>{});
            }
            // Optimization 2: If source max value < target stride, source can't reach this target
            // This is only valid when source has bounded size (is a Module)
            else if constexpr (source_is_bounded && source_max_value < target_stride) {
                return apply_to_matching_impl<SourceDim, RestInfos...>(
                    d, detail::tuple<RestInfos...>{});
            }
            // Fast path: strides match and source fits entirely in target
            else if constexpr (source_stride == target_stride && source_is_bounded &&
                               source_size <= target_size) {
                // No modulo needed since source values are always in range
                int step = d.get() * dim_stride;
                return step + apply_to_matching_impl<SourceDim, RestInfos...>(
                                  d, detail::tuple<RestInfos...>{});
            }
            // Fast path: strides match, may need modulo
            else if constexpr (source_stride == target_stride) {
                int step = (d.get() % target_size) * dim_stride;
                return step + apply_to_matching_impl<SourceDim, RestInfos...>(
                                  d, detail::tuple<RestInfos...>{});
            } else {
                // General case with folding arithmetic
                int base_value = d.get() * source_stride;
                int folded_value = (base_value / target_stride) % target_size;
                int step = folded_value * dim_stride;

                return step + apply_to_matching_impl<SourceDim, RestInfos...>(
                                  d, detail::tuple<RestInfos...>{});
            }
        }

        // Get all DimInfos that match a given base dimension type
        template <typename SourceDim>
        using matching_dim_infos_t =
            detail::keep_dim_infos_by_base_t<detail::tuple<DimInfos...>,
                                             detail::tuple<detail::get_base_dim_type_t<SourceDim>>>;

        // Helper to check if any dimension in Coordinates matches tensor dimensions
        template <typename... CoordDims>
        static constexpr bool any_coordinate_matches(const Coordinates<CoordDims...>&) {
            return (... || (detail::tuple_size<matching_dim_infos_t<CoordDims>>::value > 0));
        }

        // Helper to check if any dimension in a tuple matches tensor dimensions
        template <typename DimsTuple> struct any_coordinate_matches_impl;

        template <typename... CoordDims>
        struct any_coordinate_matches_impl<detail::tuple<CoordDims...>> {
            static constexpr bool value =
                (... || (detail::tuple_size<matching_dim_infos_t<CoordDims>>::value > 0));
        };

        template <typename DimsTuple>
        static constexpr bool any_coordinate_matches_v =
            any_coordinate_matches_impl<DimsTuple>::value;

        // Helper to apply coordinates sequentially, skipping non-matching dimensions
        DEVICE constexpr Cursor apply_coordinates_impl(const Coordinates<>&) const {
            return *this;
        }

        template <typename FirstDim, typename... RestDims>
        DEVICE constexpr Cursor
        apply_coordinates_impl(const Coordinates<FirstDim, RestDims...>& coords) const {
            using matching = matching_dim_infos_t<FirstDim>;

            // Only apply if this dimension matches something in the tensor
            if constexpr (detail::tuple_size<matching>::value > 0) {
                Cursor next = (*this)[coords.template get<FirstDim>()];
                if constexpr (sizeof...(RestDims) == 0) {
                    return next;
                } else {
                    auto rest = Coordinates<RestDims...>(coords.template get<RestDims>()...);
                    return next.apply_coordinates_impl(rest);
                }
            } else {
                // Skip this dimension, continue with rest
                if constexpr (sizeof...(RestDims) == 0) {
                    return *this;
                } else {
                    auto rest = Coordinates<RestDims...>(coords.template get<RestDims>()...);
                    return apply_coordinates_impl(rest);
                }
            }
        }

    public:
        /// @brief Apply a dimension to all tensor dimensions with the same base type.
        template <typename SourceDim>
        DEVICE constexpr Cursor apply_to_all_matching(SourceDim d) const {
            using matching = matching_dim_infos_t<SourceDim>;
            if constexpr (detail::tuple_size<matching>::value == 0) {
                return *this;
            } else {
                int offset_delta = apply_to_matching_impl(d, matching{});
                return Cursor(Base::get(), _offset + offset_delta);
            }
        }

        /// @brief Subscript operator for any dimension-like type, Coordinates, or type with
        /// coordinates().
        template <typename T> DEVICE constexpr Cursor operator[](T t) const {
            if constexpr (detail::is_coordinates_v<T>) {
                // Coordinates - normalize and apply all dimensions sequentially
                auto normalized = t.normalize();

                // At least one dimension must match
                static_assert(any_coordinate_matches_v<typename T::dims_tuple>,
                              "No dimensions in Coordinates match any tensor dimension");

                return apply_coordinates_impl(normalized);
            } else if constexpr (detail::has_coordinates_v<T>) {
                // Type with coordinates() - convert and apply
                return (*this)[t.coordinates()];
            } else if constexpr (detail::is_dim_like_v<T>) {
                // Single dimension - check for fast path first
                using matching = matching_dim_infos_t<T>;

                static_assert(detail::tuple_size<matching>::value > 0,
                              "Subscript dimension does not match any tensor dimension");

                if constexpr (has_single_exact_match<T>()) {
                    // Fast path: single exact match, direct offset calculation
                    constexpr int mem_stride = find_exact_match_stride<T>();
                    return Cursor(Base::get(), _offset + t.get() * mem_stride);
                } else {
                    // General case: apply to all matching dimensions
                    return apply_to_all_matching(t);
                }
            } else {
                static_assert(
                    detail::is_dim_like_v<T> || detail::is_coordinates_v<T> ||
                        detail::has_coordinates_v<T>,
                    "Subscript type must be Dim-like, Coordinates, or have coordinates()");
                return *this;
            }
        }

        /// @brief Increment this cursor in a specific dimension type.
        template <typename DimType> DEVICE Cursor& step(DimType d = 1) {
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
        template <typename T> DEVICE constexpr cursor_type operator[](T t) const {
            return cursor_type(Base::get())[t];
        }

        /// @brief Increment the cursor in a specific dimension type.
        template <typename DimType> DEVICE BaseCursor& step(DimType d = 1) {
            constexpr int stride =
                dim_traits::find_dim_info<DimType, DimInfos...>::info::module_type::stride.get();
            Base::reset(Base::get() + d.get() * stride);
            return *this;
        }
    };

    /// @brief Tensor class.
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
        static constexpr int total_size = detail::product_sizes<DimInfos...>::value;

        // Helper variable template for cleaner usage
        template <typename DimType>
        static constexpr bool has_dimension_v =
            dim_traits::has_dimension<DimType, DimInfos...>::value;

        // Allocate a tensor on the stack.
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

        // Get size for a specific dimension as it exists in the tensor.
        // Fails to compile if the exact dimension does not exist in the tensor.
        template <typename DimType> DEVICE static constexpr DimType size() {
            using info = typename dim_traits::find_dim_info<DimType, DimInfos...>::info;
            return DimType(info::module_type::size.get());
        }

        // Get the total extent of the tensor in the requested dimension's base type.
        // Returns the size of the coarsest fold, converted to the requested dimension type.
        template <typename DimType> DEVICE static constexpr DimType extent() {
            using RequestedBaseDim = detail::get_base_dim_type_t<DimType>;
            using matching_infos =
                detail::keep_dim_infos_by_base_t<detail::tuple<DimInfos...>,
                                                 detail::tuple<RequestedBaseDim>>;

            static_assert(detail::tuple_size<matching_infos>::value > 0,
                          "Requested dimension type does not match any tensor dimension");

            // Find the coarsest info for this base dimension
            using coarsest_info =
                typename detail::find_coarsest_info_for_base<RequestedBaseDim,
                                                             matching_infos>::type;
            using coarsest_dim_type = typename coarsest_info::dim_type;

            // Get size of coarsest dimension and convert to requested type
            return DimType(size<coarsest_dim_type>());
        }

        // Get Coordinates that include the extent for each base dimension.
        DEVICE static constexpr auto extents() {
            using coarsest_infos = detail::coarsest_dim_infos_t<detail::tuple<DimInfos...>>;
            return make_sizes_from_infos(coarsest_infos{});
        }

        /// @brief Subscript operator with any dimension type or Coordinates.
        template <typename T> DEVICE constexpr cursor_type operator[](T t) const {
            return cursor_type(get())[t];
        }

        /// @brief Get a cursor at a specific offset.
        DEVICE constexpr cursor_type offset(int offset) const {
            return cursor_type(get(), offset);
        }

        /// @brief Slice method to create a view with a different offset and size in one dimension.
        template <int SliceSize, typename SliceDimType>
        DEVICE constexpr auto slice(SliceDimType slice_start) {
            using updated_infos = detail::update_dim_info_t<SliceDimType, SliceSize, DimInfos...>;
            using tensor_type =
                typename detail::tensor_type_from_dim_info_tuple<DataType,
                                                                 updated_infos>::tensor_type;
            return tensor_type((*this)[slice_start].get());
        }

        /// @brief Load data from a source cursor that points to a shared memory buffer.
        template <typename SrcCursorType> DEVICE void load(SrcCursorType src) {
            load_impl<decltype(*this), SrcCursorType, DimInfos...>(*this, src);
        }

        /// @brief Apply a custom function to each element of the tensor
        template <typename F> DEVICE void apply(F func) {
            apply_impl<F, decltype(*this), DimInfos...>(*this, func);
        }

        /// @brief Fill the tensor with zeros.
        DEVICE void zero() {
            auto zero_func = [](auto obj) { obj->zero(); };
            apply(zero_func);
        }

        /// @brief Fill the tensor with a specified value.
        template <typename Vector> DEVICE void fill(Vector value) {
            auto fill_func = [value](auto obj) { obj->fill(value); };
            apply(fill_func);
        }

        template <typename Vector> DEVICE void add(Vector value) {
            auto add_func = [value](auto obj) { obj->add(value); };
            apply(add_func);
        }

    private:
        template <typename... Infos>
        DEVICE static constexpr auto make_sizes_from_infos(detail::tuple<Infos...>) {
            return Coordinates<typename Infos::dim_type...>(
                typename Infos::dim_type(Infos::module_type::size.get())...);
        }

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
