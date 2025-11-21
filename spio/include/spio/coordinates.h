#ifndef SPIO_COORDINATES_H_
#define SPIO_COORDINATES_H_

#include "spio/meta.h"

namespace spio {

    // Forward declarations
    template <typename... Dims> struct Coordinates;
    template <typename... DimInfos> class CompoundIndex;
    template <typename DataType, typename... DimInfos> class Tensor;

    // Forward declare make_coordinates so it can be used in detail namespace
    template <typename... Dims> DEVICE constexpr auto make_coordinates(Dims&&... dims);

    namespace detail {
        // Empty coordinates marker type - used when no dimensions match
        struct EmptyCoordinates {
            DEVICE static constexpr int num_dims() {
                return 0;
            }

            template <typename TensorOrCursor>
            DEVICE constexpr TensorOrCursor apply_to(TensorOrCursor target) const {
                return target;
            }
        };

        // Helper to apply folded coordinates to a target
        template <typename FoldedCoords, typename TensorOrCursor>
        DEVICE constexpr auto apply_folded_coords(const FoldedCoords& folded,
                                                  TensorOrCursor target) {
            if constexpr (is_same<FoldedCoords, EmptyCoordinates>::value) {
                return target;
            } else {
                return folded.apply_folded_to(target);
            }
        }

        // Get element from Coordinates by base dim type
        template <typename BaseDimType, typename... Dims>
        DEVICE constexpr auto& coords_get_by_base_dim(Coordinates<Dims...>& coords) {
            return tuple_get_by_base_dim<BaseDimType, tuple<Dims...>>::get(coords.values);
        }

        template <typename BaseDimType, typename... Dims>
        DEVICE constexpr const auto& coords_get_by_base_dim(const Coordinates<Dims...>& coords) {
            return tuple_get_by_base_dim<BaseDimType, tuple<Dims...>>::get(coords.values);
        }

        // Compare a single dimension by base type: return true if not in b, or if comparison
        // succeeds
        template <typename DimType, typename Compare, typename... ADims, typename... BDims>
        DEVICE constexpr bool compare_dim(const Coordinates<ADims...>& a,
                                          const Coordinates<BDims...>& b, Compare cmp) {
            using base_dim = get_base_dim_type_t<DimType>;
            if constexpr (tuple_contains_base_dim<base_dim, tuple<BDims...>>::value) {
                return cmp(a.template get<DimType>(), coords_get_by_base_dim<base_dim>(b));
            } else {
                return true; // No matching dim in b, condition is vacuously true
            }
        }

        // Compare already-normalized coordinates
        template <typename Compare, typename... ADims, typename... BDims>
        DEVICE constexpr bool compare_normalized_impl(const Coordinates<ADims...>& a,
                                                      const Coordinates<BDims...>& b, Compare cmp) {
            return (compare_dim<ADims>(a, b, cmp) && ...);
        }

        // Check if all shared dimensions satisfy the comparison
        // Normalizes both coordinates before comparing
        template <typename Compare, typename... ADims, typename... BDims>
        DEVICE constexpr bool compare_impl(const Coordinates<ADims...>& a,
                                           const Coordinates<BDims...>& b, Compare cmp) {
            auto na = a.normalize();
            auto nb = b.normalize();
            return compare_normalized_impl(na, nb, cmp);
        }

        // Sum dimensions matching a base type using fold expression
        // MatchingDims contains only dimensions that match BaseDimType
        template <typename... Dims, typename... MatchingDims>
        DEVICE constexpr auto sum_matching_dims(const Coordinates<Dims...>& coords,
                                                tuple<MatchingDims...>) {
            static_assert(sizeof...(MatchingDims) > 0,
                          "No dimensions match the requested base type");
            return (coords.template get<MatchingDims>() + ...);
        }

        template <typename BaseDimType, typename... Dims>
        DEVICE constexpr auto sum_dims_by_base(const Coordinates<Dims...>& coords) {
            using matching_dims = tuple_keep_base_dims_t<tuple<Dims...>, tuple<BaseDimType>>;
            return sum_matching_dims(coords, matching_dims{});
        }

        // Build normalized coordinates from unique base dims
        template <typename... BaseDims, typename... OrigDims>
        DEVICE constexpr auto build_normalized(tuple<BaseDims...>,
                                               const Coordinates<OrigDims...>& coords) {
            return make_coordinates(sum_dims_by_base<BaseDims>(coords)...);
        }

        // Process dims from A: add if B has same base dim, otherwise copy from A
        template <typename DimType, typename... ADims, typename... BDims>
        DEVICE constexpr auto process_from_a_normalized(const Coordinates<ADims...>& a,
                                                        const Coordinates<BDims...>& b) {
            using base_dim = get_base_dim_type_t<DimType>;
            if constexpr (tuple_contains_base_dim<base_dim, tuple<BDims...>>::value) {
                return a.template get<DimType>() + coords_get_by_base_dim<base_dim>(b);
            } else {
                return a.template get<DimType>();
            }
        }

        // Build result from both normalized coordinates
        template <typename... ADims, typename... BDims, typename... BOnlyDims>
        DEVICE constexpr auto add_normalized_coordinates_impl(const Coordinates<ADims...>& a,
                                                              const Coordinates<BDims...>& b,
                                                              tuple<BOnlyDims...>) {
            return make_coordinates(process_from_a_normalized<ADims>(a, b)...,
                                    b.template get<BOnlyDims>()...);
        }

        // Add two normalized coordinates
        template <typename... ADims, typename... BDims>
        DEVICE constexpr auto add_normalized_coordinates(const Coordinates<ADims...>& a,
                                                         const Coordinates<BDims...>& b) {
            using b_only = tuple_exclude_base_dims_t<tuple<BDims...>, tuple<ADims...>>;
            return add_normalized_coordinates_impl(a, b, b_only{});
        }

        // Extract DimInfos tuple from a Tensor
        template <typename T> struct get_tensor_dim_infos;

        template <template <typename, typename...> class Container, typename DataType,
                  typename... DimInfos>
        struct get_tensor_dim_infos<Container<DataType, DimInfos...>> {
            using type = tuple<DimInfos...>;
            using dim_types = tuple<typename DimInfos::dim_type...>;
        };

        template <typename T> using get_tensor_dim_infos_t = typename get_tensor_dim_infos<T>::type;

        template <typename T>
        using get_tensor_dim_types_t = typename get_tensor_dim_infos<T>::dim_types;

        // Filter DimInfos to keep only those with matching base dims
        template <typename DimInfoTuple, typename BaseDimsTuple> struct keep_dim_infos_by_base;

        template <typename BaseDimsTuple> struct keep_dim_infos_by_base<tuple<>, BaseDimsTuple> {
            using type = tuple<>;
        };

        template <typename FirstDimInfo, typename... RestDimInfos, typename BaseDimsTuple>
        struct keep_dim_infos_by_base<tuple<FirstDimInfo, RestDimInfos...>, BaseDimsTuple> {
            using dim_type = typename FirstDimInfo::dim_type;
            using base_dim = get_base_dim_type_t<dim_type>;
            static constexpr bool keep = tuple_contains<base_dim, BaseDimsTuple>::value;

            using rest =
                typename keep_dim_infos_by_base<tuple<RestDimInfos...>, BaseDimsTuple>::type;
            using type = conditional_t<keep, tuple_prepend_t<FirstDimInfo, rest>, rest>;
        };

        template <typename DimInfoTuple, typename BaseDimsTuple>
        using keep_dim_infos_by_base_t =
            typename keep_dim_infos_by_base<DimInfoTuple, BaseDimsTuple>::type;

        // Get base dims from a Coordinates dims tuple
        template <typename DimsTuple> struct get_coords_base_dims;

        template <typename... Dims> struct get_coords_base_dims<tuple<Dims...>> {
            using type = tuple<get_base_dim_type_t<Dims>...>;
        };

        template <typename DimsTuple>
        using get_coords_base_dims_t = typename get_coords_base_dims<DimsTuple>::type;

        // Convert a coordinate value to match a target DimInfo
        // The dimension type's stride is used for folding (e.g., Fold<I, 8> has stride 8)
        // The DimInfo's size is used for modulo (to keep within valid range)
        template <typename TargetDimInfo, typename SourceDim>
        DEVICE constexpr auto convert_to_dim_info(SourceDim source) {
            using TargetDimType = typename TargetDimInfo::dim_type;
            constexpr int source_stride = get_dim_stride<SourceDim>::value;
            constexpr int target_stride = get_dim_stride<TargetDimType>::value;
            constexpr int target_size = TargetDimInfo::module_type::size.get();

            int base_value = source.get() * source_stride;
            int refolded = (base_value / target_stride) % target_size;

            return TargetDimType(refolded);
        }

        // Get folded dim value using DimInfo
        template <typename TargetDimInfo, typename NormalizedCoords>
        DEVICE constexpr auto get_folded_dim_from_info(const NormalizedCoords& normalized) {
            using TargetDimType = typename TargetDimInfo::dim_type;
            using target_base = get_base_dim_type_t<TargetDimType>;
            auto source_value = coords_get_by_base_dim<target_base>(normalized);
            return convert_to_dim_info<TargetDimInfo>(source_value);
        }

        // Build folded coordinates using DimInfos - empty case
        template <typename NormalizedCoords>
        DEVICE constexpr auto build_folded_from_dim_infos(tuple<>, const NormalizedCoords&) {
            return EmptyCoordinates{};
        }

        // Build folded coordinates using DimInfos - non-empty case
        template <typename FirstDimInfo, typename... RestDimInfos, typename NormalizedCoords>
        DEVICE constexpr auto build_folded_from_dim_infos(tuple<FirstDimInfo, RestDimInfos...>,
                                                          const NormalizedCoords& normalized) {
            return make_coordinates(get_folded_dim_from_info<FirstDimInfo>(normalized),
                                    get_folded_dim_from_info<RestDimInfos>(normalized)...);
        }
    }

    /// @brief represents a set of indices for multiple dimensions
    /// @tparam Dims The dimension types contained in these coordinates
    /// @details Coordinates can hold multiple dimension values, each of a different
    ///          dimension type. Coordinates support getting dimension values by type,
    ///          normalizing to coalesce multiple dimensions with the same base type, adding
    ///          coordinates together, and applying the coordinates to tensors or cursors.
    ///         Coordinates can also be folded to match the dimension layout of a target tensor.
    ///         Coordinates implement comparison and arithmetic operators. These use the concept
    ///         of dimensional projection. When adding two Coordinates objects, dimensions with
    ///         the same base type are added together, while dimensions unique to either object
    ///         are carried over unchanged. When comparing two Coordinates objects, only shared
    ///         dimensions are compared; unique dimensions are ignored.
    template <typename... Dims> struct Coordinates {
        static_assert(sizeof...(Dims) > 0, "Coordinates must have at least one dimension.");

        // Type alias for the dimension types tuple
        using dims_tuple = detail::tuple<Dims...>;

        detail::tuple<Dims...> values;

        DEVICE constexpr Coordinates(Dims... dims) : values(dims...) {}

        template <typename DimType> DEVICE constexpr DimType& get() {
            static_assert(detail::tuple_contains<DimType, detail::tuple<Dims...>>::value,
                          "Requested DimType is not part of these Coordinates.");
            return detail::tuple_get_by_type<DimType>(values);
        }

        template <typename DimType> DEVICE constexpr const DimType& get() const {
            static_assert(detail::tuple_contains<DimType, detail::tuple<Dims...>>::value,
                          "Requested DimType is not part of these Coordinates.");
            return detail::tuple_get_by_type<DimType>(values);
        }

        DEVICE constexpr auto normalize() const {
            using unique_bases = detail::tuple_unique_base_dims_t<detail::tuple<Dims...>>;
            return detail::build_normalized(unique_bases{}, *this);
        }

        template <typename... OtherDims>
        DEVICE constexpr auto operator+(const Coordinates<OtherDims...>& other) const {
            return detail::add_normalized_coordinates(normalize(), other.normalize());
        }

        // Add Coordinates + CompoundIndex
        template <typename... DimInfos>
        DEVICE constexpr auto operator+(const CompoundIndex<DimInfos...>& idx) const {
            return *this + idx.to_coordinates();
        }

        template <typename... OtherDims>
        DEVICE constexpr bool operator<(const Coordinates<OtherDims...>& other) const {
            return detail::compare_impl(*this, other,
                                        [](const auto& a, const auto& b) { return a < b; });
        }

        template <typename... OtherDims>
        DEVICE constexpr bool operator<=(const Coordinates<OtherDims...>& other) const {
            return detail::compare_impl(*this, other,
                                        [](const auto& a, const auto& b) { return a <= b; });
        }

        template <typename... OtherDims>
        DEVICE constexpr bool operator>(const Coordinates<OtherDims...>& other) const {
            return detail::compare_impl(*this, other,
                                        [](const auto& a, const auto& b) { return a > b; });
        }

        template <typename... OtherDims>
        DEVICE constexpr bool operator>=(const Coordinates<OtherDims...>& other) const {
            return detail::compare_impl(*this, other,
                                        [](const auto& a, const auto& b) { return a >= b; });
        }

        template <typename... OtherDims>
        DEVICE constexpr bool operator==(const Coordinates<OtherDims...>& other) const {
            return detail::compare_impl(*this, other,
                                        [](const auto& a, const auto& b) { return a == b; });
        }

        template <typename... OtherDims>
        DEVICE constexpr bool operator!=(const Coordinates<OtherDims...>& other) const {
            return !(*this == other);
        }

        // Comparison with CompoundIndex - converts CompoundIndex to Coordinates first
        template <typename... DimInfos>
        DEVICE constexpr bool operator<(const CompoundIndex<DimInfos...>& idx) const {
            return *this < idx.to_coordinates();
        }

        template <typename... DimInfos>
        DEVICE constexpr bool operator<=(const CompoundIndex<DimInfos...>& idx) const {
            return *this <= idx.to_coordinates();
        }

        template <typename... DimInfos>
        DEVICE constexpr bool operator>(const CompoundIndex<DimInfos...>& idx) const {
            return *this > idx.to_coordinates();
        }

        template <typename... DimInfos>
        DEVICE constexpr bool operator>=(const CompoundIndex<DimInfos...>& idx) const {
            return *this >= idx.to_coordinates();
        }

        template <typename... DimInfos>
        DEVICE constexpr bool operator==(const CompoundIndex<DimInfos...>& idx) const {
            return *this == idx.to_coordinates();
        }

        template <typename... DimInfos>
        DEVICE constexpr bool operator!=(const CompoundIndex<DimInfos...>& idx) const {
            return *this != idx.to_coordinates();
        }

        /// @brief Apply the coordinates to a tensor or cursor.
        /// @details First folds the coordinates to match the target's dimension layout,
        ///          then applies every matching dimension to the tensor or cursor's
        ///          apply_index_if_found method.
        /// @tparam TensorOrCursor the type of the tensor or cursor to apply the coordinates to
        /// @param target the tensor or cursor to apply the coordinates to
        /// @return a new cursor with the coordinates applied
        template <typename TensorOrCursor>
        DEVICE constexpr auto apply_to(TensorOrCursor target) const {
            auto folded = fold_like<TensorOrCursor>();
            return detail::apply_folded_coords(folded, target);
        }

        /// @brief Get the number of dimensions in these coordinates.
        DEVICE static constexpr int num_dims() {
            return sizeof...(Dims);
        }

        /// @brief Fold the coordinates to match the dimension layout of a tensor or cursor.
        /// @details For each dimension in the target tensor/cursor that has a matching base
        ///          dimension type in these coordinates, creates a new coordinate value
        ///          folded to match the target's stride.
        /// @tparam TensorOrCursor the type of the tensor or cursor to fold like
        /// @param target the tensor or cursor whose dimension layout to match (used for type only)
        /// @return a new Coordinates object with dimensions folded to match the target
        template <typename TensorOrCursor>
        DEVICE constexpr auto fold_like(const TensorOrCursor& target) const {
            (void)target;
            return fold_like<TensorOrCursor>();
        }

        template <typename TensorOrCursor> DEVICE constexpr auto fold_like() const {
            // Get all DimInfos from the target tensor
            using all_dim_infos = detail::get_tensor_dim_infos_t<TensorOrCursor>;
            // Get base dims from our coordinates
            using coords_base_dims = detail::get_coords_base_dims_t<dims_tuple>;
            // Keep only DimInfos that match our base dims
            using matching_dim_infos =
                detail::keep_dim_infos_by_base_t<all_dim_infos, coords_base_dims>;

            auto normalized = normalize();
            return detail::build_folded_from_dim_infos(matching_dim_infos{}, normalized);
        }

        // Apply already-folded coordinates to a target
        template <typename TensorOrCursor>
        DEVICE constexpr auto apply_folded_to(TensorOrCursor target) const {
            return apply_dimensions<TensorOrCursor, Dims...>(target);
        }

    private:
        // Base case: no more dimensions to apply
        template <typename TensorOrCursor>
        DEVICE constexpr auto apply_dimensions(TensorOrCursor target) const {
            return target;
        }

        // Recursive case: apply the first dimension, then recurse with the rest
        template <typename TensorOrCursor, typename FirstDim, typename... RestDims>
        DEVICE constexpr auto apply_dimensions(TensorOrCursor target) const {
            auto dim_value = get<FirstDim>();
            auto next_target = target.apply_index_if_found(dim_value);
            return apply_dimensions<decltype(next_target), RestDims...>(next_target);
        }
    };

    template <typename... Dims> DEVICE constexpr auto make_coordinates(Dims&&... dims) {
        return Coordinates<detail::decay_t<Dims>...>(static_cast<Dims&&>(dims)...);
    }

    template <typename... Dims> DEVICE constexpr auto make_normalized_coordinates(Dims&&... dims) {
        return make_coordinates(static_cast<Dims&&>(dims)...).normalize();
    }

    /// @brief Iterator for iterating over all coordinate combinations using CompoundIndex
    template <typename IndexType> class CoordinatesIterator {
    public:
        using coordinates_type = decltype(IndexType(0).to_coordinates());

        DEVICE constexpr CoordinatesIterator(int offset) : _offset(offset) {}

        DEVICE constexpr coordinates_type operator*() const {
            return IndexType(_offset).to_coordinates();
        }

        DEVICE constexpr CoordinatesIterator& operator++() {
            ++_offset;
            return *this;
        }

        DEVICE constexpr bool operator!=(const CoordinatesIterator& other) const {
            return _offset != other._offset;
        }

        DEVICE constexpr bool operator==(const CoordinatesIterator& other) const {
            return _offset == other._offset;
        }

    private:
        int _offset;
    };

    /// @brief Range for iterating over all coordinate combinations using CompoundIndex
    template <typename IndexType> class CoordinatesRange {
    public:
        using iterator = CoordinatesIterator<IndexType>;
        using coordinates_type = decltype(IndexType(0).to_coordinates());

        constexpr CoordinatesRange() = default;

        DEVICE constexpr iterator begin() const {
            return iterator(0);
        }

        DEVICE constexpr iterator end() const {
            return iterator(IndexType::total_size);
        }
    };

    /// @brief Create a range that iterates over all coordinates of a tensor
    template <typename DataType, typename... DimInfos>
    DEVICE constexpr auto range(const Tensor<DataType, DimInfos...>&) {
        return CoordinatesRange<CompoundIndex<DimInfos...>>();
    }
}

#endif