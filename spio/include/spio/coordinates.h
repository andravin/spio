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

        // Note: keep_dim_infos_by_base is defined in meta.h

        // ========================================================================
        // Coordinates helpers - forward declarations for mutual dependencies
        // ========================================================================

        // Get element from Coordinates by base dim type
        template <typename BaseDimType, typename... Dims>
        DEVICE constexpr auto& coords_get_by_base_dim(Coordinates<Dims...>& coords);

        template <typename BaseDimType, typename... Dims>
        DEVICE constexpr const auto& coords_get_by_base_dim(const Coordinates<Dims...>& coords);

        // Detect if T is a Coordinates type
        template <typename T> struct is_coordinates : false_type {};

        template <typename... Dims> struct is_coordinates<Coordinates<Dims...>> : true_type {};

        template <typename T> inline constexpr bool is_coordinates_v = is_coordinates<T>::value;

        // Detect if T has a coordinates() method
        template <typename T, typename = void> struct has_coordinates : false_type {};

        template <typename T>
        struct has_coordinates<T, void_t<decltype(declval<T>().coordinates())>> : true_type {};

        template <typename T> inline constexpr bool has_coordinates_v = has_coordinates<T>::value;

        // ========================================================================
        // Validation helpers
        // ========================================================================

        // Check if two Coordinates have any base dimensions in common
        template <typename CoordsTuple1, typename CoordsTuple2> struct coords_share_base_dim;

        template <typename CoordsTuple2> struct coords_share_base_dim<tuple<>, CoordsTuple2> {
            static constexpr bool value = false;
        };

        template <typename First, typename... Rest, typename CoordsTuple2>
        struct coords_share_base_dim<tuple<First, Rest...>, CoordsTuple2> {
            using first_base = typename First::dim_type;
            static constexpr bool value =
                tuple_contains_base_dim<first_base, CoordsTuple2>::value ||
                coords_share_base_dim<tuple<Rest...>, CoordsTuple2>::value;
        };

        template <typename CoordsTuple1, typename CoordsTuple2>
        inline constexpr bool coords_share_base_dim_v =
            coords_share_base_dim<CoordsTuple1, CoordsTuple2>::value;

        // ========================================================================
        // Comparison helpers
        // ========================================================================

        // Compare a single dimension by base type: return true if not in b, or if comparison
        // succeeds
        template <typename DimType, typename Compare, typename... ADims, typename... BDims>
        DEVICE constexpr bool compare_dim(const Coordinates<ADims...>& a,
                                          const Coordinates<BDims...>& b, Compare cmp) {
            using base_dim = typename DimType::dim_type;
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
            static_assert(coords_share_base_dim_v<tuple<ADims...>, tuple<BDims...>>,
                          "Coordinates have no base dimensions in common");
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

        // ========================================================================
        // Normalization helpers
        // ========================================================================

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

        // ========================================================================
        // Addition helpers
        // ========================================================================

        // Process dims from A: add if B has same base dim, otherwise copy from A
        template <typename DimType, typename... ADims, typename... BDims>
        DEVICE constexpr auto process_from_a_normalized(const Coordinates<ADims...>& a,
                                                        const Coordinates<BDims...>& b) {
            using base_dim = typename DimType::dim_type;
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

        // Concatenate two coordinate sets without normalization.
        template <typename... ADims, typename... BDims>
        DEVICE constexpr auto concat_coordinates(const Coordinates<ADims...>& a,
                                                 const Coordinates<BDims...>& b) {
            return make_coordinates(a.template get<ADims>()..., b.template get<BDims>()...);
        }

        // ========================================================================
        // coords_get_by_base_dim implementations
        // ========================================================================

        template <typename BaseDimType, typename... Dims>
        DEVICE constexpr auto& coords_get_by_base_dim(Coordinates<Dims...>& coords) {
            return tuple_get_by_base_dim<BaseDimType, tuple<Dims...>>::get(coords.values);
        }

        template <typename BaseDimType, typename... Dims>
        DEVICE constexpr const auto& coords_get_by_base_dim(const Coordinates<Dims...>& coords) {
            return tuple_get_by_base_dim<BaseDimType, tuple<Dims...>>::get(coords.values);
        }

    } // end detail namespace

    // ============================================================================
    // Empty Coordinates specialization (for base case of sequential apply)
    // ============================================================================
    template <> struct Coordinates<> {
        using dims_tuple = detail::tuple<>;
        detail::tuple<> values;

        DEVICE constexpr Coordinates() : values() {}

        DEVICE constexpr auto normalize() const {
            return *this;
        }

        DEVICE static constexpr int num_dims() {
            return 0;
        }

        template <typename... OtherDims>
        DEVICE constexpr auto operator+(const Coordinates<OtherDims...>& other) const {
            return other;
        }

        /// @brief Add a single dimension to empty coordinates.
        template <typename Dim> DEVICE constexpr auto operator+(Dim d) const {
            static_assert(detail::is_dim_like_v<Dim>, "Argument must be a dimension type");
            return make_coordinates(d);
        }
    };

    /// @brief represents a set of indices for multiple dimensions
    /// @tparam Dims The dimension types contained in these coordinates
    template <typename... Dims> struct Coordinates {
        static_assert(sizeof...(Dims) > 0, "Coordinates must have at least one dimension.");

        using dims_tuple = detail::tuple<Dims...>;

        detail::tuple<Dims...> values;

        /// @brief Default constructor - initializes all dimensions to zero.
        DEVICE constexpr Coordinates() : values(Dims(0)...) {}

        DEVICE constexpr Coordinates(Dims... dims) : values(dims...) {}

        /// @brief Get the value of a specific dimension by exact type.
        template <typename DimType> DEVICE constexpr auto& get() {
            static_assert(detail::tuple_contains<DimType, detail::tuple<Dims...>>::value,
                          "Requested DimType is not part of these Coordinates.");
            return detail::tuple_get_by_type<DimType>(values);
        }

        /// @brief Get the value of a specific dimension by exact type (const version).
        template <typename DimType> DEVICE constexpr const auto& get() const {
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

        /// @brief Add a type with coordinates() method (e.g., CompoundIndex types).
        template <
            typename T,
            detail::enable_if_t<detail::has_coordinates_v<T> && !detail::is_dim_like_v<T>, int> = 0>
        DEVICE constexpr auto operator+(const T& t) const {
            return *this + t.coordinates();
        }

        /// @brief Add a single dimension to coordinates.
        template <typename Dim,
                  detail::enable_if_t<detail::is_dim_like_v<Dim> && !detail::has_coordinates_v<Dim>,
                                      int> = 0>
        DEVICE constexpr auto operator+(Dim d) const {
            return *this + make_coordinates(d);
        }

        /// @brief Compare coordinates with a single dimension.
        template <typename Dim, detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
        DEVICE constexpr bool operator<(Dim d) const {
            return *this < make_coordinates(d);
        }

        template <typename Dim, detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
        DEVICE constexpr bool operator<=(Dim d) const {
            return *this <= make_coordinates(d);
        }

        template <typename Dim, detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
        DEVICE constexpr bool operator>(Dim d) const {
            return *this > make_coordinates(d);
        }

        template <typename Dim, detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
        DEVICE constexpr bool operator>=(Dim d) const {
            return *this >= make_coordinates(d);
        }

        template <typename Dim, detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
        DEVICE constexpr bool operator==(Dim d) const {
            return *this == make_coordinates(d);
        }

        template <typename Dim, detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
        DEVICE constexpr bool operator!=(Dim d) const {
            return *this != make_coordinates(d);
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

        /// @brief Compare with a type that has coordinates() method (e.g., CompoundIndex types).
        template <
            typename T,
            detail::enable_if_t<detail::has_coordinates_v<T> && !detail::is_dim_like_v<T>, int> = 0>
        DEVICE constexpr bool operator<(const T& t) const {
            return *this < t.coordinates();
        }

        template <
            typename T,
            detail::enable_if_t<detail::has_coordinates_v<T> && !detail::is_dim_like_v<T>, int> = 0>
        DEVICE constexpr bool operator<=(const T& t) const {
            return *this <= t.coordinates();
        }

        template <
            typename T,
            detail::enable_if_t<detail::has_coordinates_v<T> && !detail::is_dim_like_v<T>, int> = 0>
        DEVICE constexpr bool operator>(const T& t) const {
            return *this > t.coordinates();
        }

        template <
            typename T,
            detail::enable_if_t<detail::has_coordinates_v<T> && !detail::is_dim_like_v<T>, int> = 0>
        DEVICE constexpr bool operator>=(const T& t) const {
            return *this >= t.coordinates();
        }

        template <
            typename T,
            detail::enable_if_t<detail::has_coordinates_v<T> && !detail::is_dim_like_v<T>, int> = 0>
        DEVICE constexpr bool operator==(const T& t) const {
            return *this == t.coordinates();
        }

        template <
            typename T,
            detail::enable_if_t<detail::has_coordinates_v<T> && !detail::is_dim_like_v<T>, int> = 0>
        DEVICE constexpr bool operator!=(const T& t) const {
            return *this != t.coordinates();
        }

        /// @brief Get the number of dimensions in these coordinates.
        DEVICE static constexpr int num_dims() {
            return sizeof...(Dims);
        }
    };

    template <typename... Dims> DEVICE constexpr auto make_coordinates(Dims&&... dims) {
        return Coordinates<detail::decay_t<Dims>...>(static_cast<Dims&&>(dims)...);
    }

    template <typename... Dims> DEVICE constexpr auto make_normalized_coordinates(Dims&&... dims) {
        return make_coordinates(static_cast<Dims&&>(dims)...).normalize();
    }

    // ============================================================================
    // Free function operators for Dim + Coordinates and Dim <op> Coordinates
    // ============================================================================

    namespace detail {
        // Check if two dim-like types have orthogonal (different) base dimensions.
        // This is true when both types have dim_type but they are different.
        template <typename T, typename U, typename = void>
        struct is_orthogonal_dims : false_type {};

        template <typename T, typename U>
        struct is_orthogonal_dims<T, U, void_t<typename T::dim_type, typename U::dim_type>>
            : bool_constant<is_dim_like_v<T> && is_dim_like_v<U> &&
                            !is_same<typename T::dim_type, typename U::dim_type>::value> {};

        template <typename T, typename U>
        inline constexpr bool is_orthogonal_dims_v = is_orthogonal_dims<T, U>::value;
    } // namespace detail

    // ============================================================================
    // Orthogonal dimension operators: different dim types produce Coordinates
    // ============================================================================

    /// @brief Add two orthogonal dimensions to produce Coordinates (e.g., I(1) + J(2))
    template <typename T, typename U,
              detail::enable_if_t<detail::is_orthogonal_dims_v<T, U>, int> = 0>
    DEVICE constexpr auto operator+(T lhs, U rhs) {
        return make_coordinates(lhs, rhs);
    }

    /// @brief Subtract two orthogonal dimensions to produce Coordinates (e.g., I(1) - J(2))
    /// The second dimension is negated: I(1) - J(2) == make_coordinates(I(1), J(-2))
    template <typename T, typename U,
              detail::enable_if_t<detail::is_orthogonal_dims_v<T, U>, int> = 0>
    DEVICE constexpr auto operator-(T lhs, U rhs) {
        return make_coordinates(lhs, U(-rhs.get()));
    }

    /// @brief Add a dimension to coordinates (Dim + Coordinates)
    template <typename Dim, typename... CoordDims,
              detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
    DEVICE constexpr auto operator+(Dim d, const Coordinates<CoordDims...>& coords) {
        return make_coordinates(d) + coords;
    }

    /// @brief Compare dimension with coordinates (Dim < Coordinates)
    template <typename Dim, typename... CoordDims,
              detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
    DEVICE constexpr bool operator<(Dim d, const Coordinates<CoordDims...>& coords) {
        return make_coordinates(d) < coords;
    }

    /// @brief Compare dimension with coordinates (Dim <= Coordinates)
    template <typename Dim, typename... CoordDims,
              detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
    DEVICE constexpr bool operator<=(Dim d, const Coordinates<CoordDims...>& coords) {
        return make_coordinates(d) <= coords;
    }

    /// @brief Compare dimension with coordinates (Dim > Coordinates)
    template <typename Dim, typename... CoordDims,
              detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
    DEVICE constexpr bool operator>(Dim d, const Coordinates<CoordDims...>& coords) {
        return make_coordinates(d) > coords;
    }

    /// @brief Compare dimension with coordinates (Dim >= Coordinates)
    template <typename Dim, typename... CoordDims,
              detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
    DEVICE constexpr bool operator>=(Dim d, const Coordinates<CoordDims...>& coords) {
        return make_coordinates(d) >= coords;
    }

    /// @brief Compare dimension with coordinates (Dim == Coordinates)
    template <typename Dim, typename... CoordDims,
              detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
    DEVICE constexpr bool operator==(Dim d, const Coordinates<CoordDims...>& coords) {
        return make_coordinates(d) == coords;
    }

    /// @brief Compare dimension with coordinates (Dim != Coordinates)
    template <typename Dim, typename... CoordDims,
              detail::enable_if_t<detail::is_dim_like_v<Dim>, int> = 0>
    DEVICE constexpr bool operator!=(Dim d, const Coordinates<CoordDims...>& coords) {
        return make_coordinates(d) != coords;
    }

    /// @brief Iterator for iterating over all coordinate combinations using CompoundIndex
    template <typename IndexType> class CoordinatesIterator {
    public:
        using coordinates_type = decltype(IndexType(0).coordinates());

        DEVICE constexpr CoordinatesIterator(int offset) : _offset(offset) {}

        DEVICE constexpr coordinates_type operator*() const {
            return IndexType(_offset).coordinates();
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
        using coordinates_type = decltype(IndexType(0).coordinates());

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

    /// @brief Create a range that iterates over all fragment coordinates
    /// Works with any fragment type that has a tensor_type alias
    /// Exclude dim-like types to avoid ambiguity with range(dim_type)
    template <typename FragmentType, detail::enable_if_t<detail::has_tensor_type_v<FragmentType> &&
                                                             !detail::is_dim_like_v<FragmentType>,
                                                         int> = 0>
    DEVICE constexpr auto range(const FragmentType&) {
        using tensor_t = typename FragmentType::tensor_type;
        return range(tensor_t(nullptr));
    }

}

#endif