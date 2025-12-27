#ifndef SPIO_DIM_INFO_H
#define SPIO_DIM_INFO_H

#include "spio/macros.h"
#include "spio/dim.h"
#include "spio/meta.h"

namespace spio {
    namespace detail {
        // Helper template to compute product of sizes at compile time
        template <typename... Ts> struct product_sizes;

        template <typename T, typename... Ts> struct product_sizes<T, Ts...> {
            static constexpr int value = T::size * product_sizes<Ts...>::value;
        };

        template <typename T> struct product_sizes<T> {
            static constexpr int value = T::size;
        };

        // Add a new trait to identify dummy dimensions
        template <typename DimInfo> struct is_dummy_dimension {
            static constexpr bool value = false;
        };
    }

    template <typename DimType, int Size> struct DimSize {
        using dim_type = DimType;
        static constexpr int size = Size;

        static constexpr auto max_value = DimType(Size - 1);
    };

    /// @brief Store information about a tensor dimension.
    /// @tparam DimType the dimension type
    /// @tparam Size the size of the dimension
    /// @tparam Stride the stride of the dimension
    template <typename DimType, int Size, int Stride> struct DimInfo : DimSize<DimType, Size> {
        using Base = DimSize<DimType, Size>;
        using Base::dim_type;
        using Base::size;
        static constexpr int stride = Stride;

        DEVICE static constexpr int dim_to_offset(DimType dim) {
            return (dim.get() % size) * stride;
        }

        DEVICE static constexpr auto offset_to_dim(unsigned offset) {
            using dim_base_type = typename DimType::dim_type;
            constexpr int dim_stride = DimType::stride;
            // Unsigned arithmetic generates smaller code for division/modulus.
            constexpr unsigned usize = size;
            constexpr unsigned ustride = stride;
            auto value = (offset / ustride) % usize;
            return Module<dim_base_type, size, dim_stride>(value);
        }
    };

    namespace detail {
        // Type trait to detect DimInfo types
        template <typename T> struct is_dim_info : false_type {};

        template <typename DimType, int Size, int Stride>
        struct is_dim_info<DimInfo<DimType, Size, Stride>> : true_type {};

        template <typename T> inline constexpr bool is_dim_info_v = is_dim_info<T>::value;

        // Type trait to detect derived dimension types (have input_dims and output_dims)
        template <typename T, typename = void> struct is_derived_dim : false_type {};

        template <typename T>
        struct is_derived_dim<T, void_t<typename T::input_dims, typename T::output_dims>>
            : true_type {};

        template <typename T> inline constexpr bool is_derived_dim_v = is_derived_dim<T>::value;

        // Filter a parameter pack to keep only DimInfo types
        template <typename... Ts> struct filter_dim_infos;

        template <> struct filter_dim_infos<> {
            using type = tuple<>;
        };

        template <typename First, typename... Rest> struct filter_dim_infos<First, Rest...> {
            using rest_result = typename filter_dim_infos<Rest...>::type;
            using type = conditional_t<is_dim_info_v<First>, tuple_prepend_t<First, rest_result>,
                                       rest_result>;
        };

        template <typename... Ts> using filter_dim_infos_t = typename filter_dim_infos<Ts...>::type;

        // Filter a parameter pack to keep only derived dimension types
        template <typename... Ts> struct filter_derived_dims;

        template <> struct filter_derived_dims<> {
            using type = tuple<>;
        };

        template <typename First, typename... Rest> struct filter_derived_dims<First, Rest...> {
            using rest_result = typename filter_derived_dims<Rest...>::type;
            using type = conditional_t<is_derived_dim_v<First>, tuple_prepend_t<First, rest_result>,
                                       rest_result>;
        };

        template <typename... Ts>
        using filter_derived_dims_t = typename filter_derived_dims<Ts...>::type;

        // ========================================================================
        // Derived dimension matching utilities
        // ========================================================================

        // Helper to check if max_value >= min_positive_value for same-base types
        template <typename DimSizeType, typename CoordDim, bool SameBase> struct check_value_range {
            static constexpr bool value = false; // Different base dims never match
        };

        template <typename DimSizeType, typename CoordDim>
        struct check_value_range<DimSizeType, CoordDim, true> {
            // Same base - safe to compare using base units
            // min_positive in base units = CoordDim::stride
            static constexpr bool value = DimSizeType::max_value >= CoordDim(1);
        };

        // Check if a subscript coordinate can meaningfully cover a DimSize's range.
        // The subscript's min_positive_value must not exceed the DimSize's max_value.
        //
        // Example: DimSize<K8, 2> has max_value = K8(1).
        //   - K: K8(1) >= K::min_positive_value() → match
        //   - K8: K8(1) >= K8::min_positive_value() → match
        //   - K64: K8(1) >= K64::min_positive_value() → false
        template <typename CoordDim, typename DimSizeType> struct coord_matches_dim_size {
            using coord_base = typename CoordDim::dim_type;
            using dim_size_base = typename DimSizeType::dim_type::dim_type;
            static constexpr bool same_base = is_same<coord_base, dim_size_base>::value;

            // Only compare values if bases match
            static constexpr bool value =
                same_base && check_value_range<DimSizeType, CoordDim, same_base>::value;
        };

        // Check if a single input_dim is matched by any coordinate in CoordsTuple
        template <typename DimSizeType, typename CoordsTuple> struct input_dim_is_matched;

        template <typename DimSizeType> struct input_dim_is_matched<DimSizeType, tuple<>> {
            static constexpr bool value = false;
        };

        template <typename DimSizeType, typename FirstCoord, typename... RestCoords>
        struct input_dim_is_matched<DimSizeType, tuple<FirstCoord, RestCoords...>> {
            static constexpr bool value =
                coord_matches_dim_size<FirstCoord, DimSizeType>::value ||
                input_dim_is_matched<DimSizeType, tuple<RestCoords...>>::value;
        };

        // Check if ALL input_dims of a derived_dim are matched by coordinates
        template <typename InputDimsTuple, typename CoordsTuple> struct all_input_dims_matched;

        template <typename CoordsTuple> struct all_input_dims_matched<tuple<>, CoordsTuple> {
            static constexpr bool value = true;
        };

        template <typename FirstInputDim, typename... RestInputDims, typename CoordsTuple>
        struct all_input_dims_matched<tuple<FirstInputDim, RestInputDims...>, CoordsTuple> {
            static constexpr bool value =
                input_dim_is_matched<FirstInputDim, CoordsTuple>::value &&
                all_input_dims_matched<tuple<RestInputDims...>, CoordsTuple>::value;
        };

        // Check if a derived_dim is fully matched by the coordinates
        template <typename DerivedDim, typename CoordsTuple> struct derived_dim_matches_coords {
            static constexpr bool value =
                all_input_dims_matched<typename DerivedDim::input_dims, CoordsTuple>::value;
        };

        // Find the first derived_dim that matches the coordinates (or void if none)
        template <typename DerivedDimsTuple, typename CoordsTuple> struct find_matching_derived_dim;

        template <typename CoordsTuple> struct find_matching_derived_dim<tuple<>, CoordsTuple> {
            using type = void;
            static constexpr bool found = false;
        };

        template <typename FirstDerived, typename... RestDerived, typename CoordsTuple>
        struct find_matching_derived_dim<tuple<FirstDerived, RestDerived...>, CoordsTuple> {
            static constexpr bool first_matches =
                derived_dim_matches_coords<FirstDerived, CoordsTuple>::value;
            using rest_result = find_matching_derived_dim<tuple<RestDerived...>, CoordsTuple>;
            using type = conditional_t<first_matches, FirstDerived, typename rest_result::type>;
            static constexpr bool found = first_matches || rest_result::found;
        };

        template <typename DerivedDimsTuple, typename CoordsTuple>
        using find_matching_derived_dim_t =
            typename find_matching_derived_dim<DerivedDimsTuple, CoordsTuple>::type;

        // ========================================================================
        // Project coordinates onto derived dimension input types
        // ========================================================================

        // Find the coordinate that matches a DimSize's base dimension
        template <typename DimSizeType, typename CoordsTuple> struct find_coord_for_dim_size;

        template <typename DimSizeType> struct find_coord_for_dim_size<DimSizeType, tuple<>> {
            // Should not happen if matching was done correctly
            using type = void;
        };

        template <typename DimSizeType, typename FirstCoord, typename... RestCoords>
        struct find_coord_for_dim_size<DimSizeType, tuple<FirstCoord, RestCoords...>> {
            static constexpr bool matches = coord_matches_dim_size<FirstCoord, DimSizeType>::value;
            using type = conditional_t<
                matches, FirstCoord,
                typename find_coord_for_dim_size<DimSizeType, tuple<RestCoords...>>::type>;
        };

        // Project a single coordinate onto the target DimSize's dim_type
        template <typename DimSizeType, typename CoordsTuple> struct project_coord_to_dim_size {
            using source_coord = typename find_coord_for_dim_size<DimSizeType, CoordsTuple>::type;
            using target_dim = typename DimSizeType::dim_type;
            static constexpr int target_size = DimSizeType::size;
            static constexpr int target_stride = target_dim::stride;

            template <typename Coords> DEVICE static constexpr auto project(const Coords& coords) {
                auto source_value = coords.template get<source_coord>();
                return source_value.template fold<target_stride>() % target_size;
            }
        };

        // Build input_coordinates for a derived dimension from subscript coordinates
        // We need to project each subscript coord to the corresponding input dim type
        template <typename DerivedDim, typename CoordsTuple> struct build_input_coords_helper {
            using input_dims = typename DerivedDim::input_dims;

            // Project the nth input dim from the subscript coords
            template <typename DimSizeType, typename Coords>
            DEVICE static constexpr auto project_one(const Coords& coords) {
                return project_coord_to_dim_size<DimSizeType, CoordsTuple>::project(coords);
            }

            // Build the full input_coordinates by projecting each input dim
            template <typename Coords, typename... DimSizes>
            DEVICE static constexpr auto build_from_tuple(const Coords& coords,
                                                          tuple<DimSizes...>) {
                return typename DerivedDim::input_coordinates(project_one<DimSizes>(coords)...);
            }

            template <typename Coords> DEVICE static constexpr auto build(const Coords& coords) {
                return build_from_tuple(coords, input_dims{});
            }
        };

        template <typename DerivedDim, typename Coords>
        DEVICE constexpr auto build_derived_input_coords(const Coords& coords) {
            using coords_tuple = typename Coords::dims_tuple;
            return build_input_coords_helper<DerivedDim, coords_tuple>::build(coords);
        }

        // Apply a derived dimension: project inputs, compute output, return output coords
        template <typename DerivedDim, typename Coords>
        DEVICE constexpr auto apply_derived_dim(const Coords& coords) {
            auto input_coords = build_derived_input_coords<DerivedDim>(coords);
            return DerivedDim::compute_coordinates(input_coords);
        }
    }

    namespace detail {
        /// @brief Check if a dimension exists in the tensor.
        /// @tparam DimType the dimension type to check
        /// @tparam DimInfos the dimension infos
        template <typename DimType, typename... DimInfos> struct has_dim;

        template <typename DimType, typename FirstDimInfo, typename... RestDimInfos>
        struct has_dim<DimType, FirstDimInfo, RestDimInfos...> {
            static constexpr bool value =
                detail::is_same<DimType, typename FirstDimInfo::dim_type>::value ||
                has_dim<DimType, RestDimInfos...>::value;
        };

        template <typename DimType> struct has_dim<DimType> {
            static constexpr bool value = false;
        };

        template <typename DimType, typename... DimInfos> struct find_dim_info_impl;

        template <typename DimType, typename FirstDimInfo, typename... RestDimInfos>
        struct find_dim_info_impl<DimType, FirstDimInfo, RestDimInfos...> {
            static constexpr bool is_match =
                detail::is_same<DimType, typename FirstDimInfo::dim_type>::value;
            using info =
                detail::conditional_t<is_match, FirstDimInfo,
                                      typename find_dim_info_impl<DimType, RestDimInfos...>::info>;
        };

        /// @brief Base case with dummy DimInfo instantiation for error handling.
        template <typename DimType> struct find_dim_info_impl<DimType> {
            using info = DimInfo<DimType, 0, 1>;
        };
    }

    // Type traits for dim_info operations
    namespace dim_traits {
        /// @brief Find dimension info for a given dimension type.
        /// @tparam DimType the dimension type to find
        /// @tparam ...DimInfos the dimension infos
        template <typename DimType, typename... DimInfos> struct find_dim_info {
            // First check if dimension exists and show a clear error message if it doesn't.
            static_assert(detail::has_dim<DimType, DimInfos...>::value,
                          "Dimension type not found in tensor - ensure you're using the correct "
                          "dimension type");

            // Then find the dimension info.
            using impl = detail::find_dim_info_impl<DimType, DimInfos...>;
            using info = typename impl::info;
        };

        /// @brief Check if a dimension exists in the tensor (public interface).
        /// @tparam DimType the dimension type to check
        /// @tparam DimInfos the dimension infos
        template <typename DimType, typename... DimInfos> struct has_dimension {
            static constexpr bool value = detail::has_dim<DimType, DimInfos...>::value;
        };

        /// @brief Get the size of a specific dimension.
        /// @tparam DimType the dimension type
        /// @tparam DimInfos the dimension infos
        template <typename DimType, typename... DimInfos> struct dimension_size {
            static constexpr DimType value = find_dim_info<DimType, DimInfos...>::info::size;
        };

        template <typename DimType, typename... DimInfos> struct dimension_stride {
            static constexpr DimType value = find_dim_info<DimType, DimInfos...>::info::stride;
        };

        template <typename FromDim, typename ToDim, typename DimInfo>
        struct replace_dimension_impl {
            using type = DimInfo;
        };

        template <typename FromDim, typename ToDim, typename DimType, int Size, int Stride>
        struct replace_dimension_impl<FromDim, ToDim, DimInfo<DimType, Size, Stride>> {
            static constexpr bool is_target = detail::is_same<DimType, FromDim>::value;

            using type = typename detail::conditional_t<is_target, DimInfo<ToDim, Size, Stride>,
                                                        DimInfo<DimType, Size, Stride>>;
        };

        /// @brief Replace all occurrences of a base dimension with another base dimension.
        /// @details This preserves the fold structure. For example:
        ///   - DimInfo<X, Size, Stride> -> DimInfo<Y, Size, Stride>
        ///   - DimInfo<Fold<X, 8>, Size, Stride> -> DimInfo<Fold<Y, 8>, Size, Stride>
        /// @tparam FromBaseDim the base dimension to replace
        /// @tparam ToBaseDim the base dimension to replace with
        /// @tparam DimInfoType the DimInfo to transform
        template <typename FromBaseDim, typename ToBaseDim, typename DimInfoType>
        struct replace_dimension;

        template <typename FromBaseDim, typename ToBaseDim, typename DimType, int Size, int Stride>
        struct replace_dimension<FromBaseDim, ToBaseDim, DimInfo<DimType, Size, Stride>> {
            using new_dim_type = detail::replace_base_dim_t<FromBaseDim, ToBaseDim, DimType>;
            using type = DimInfo<new_dim_type, Size, Stride>;
        };

        // ========================================================================
        // Tuple-based variants for use with dim_infos alias
        // ========================================================================

        /// @brief Check if a dimension exists in a tuple of DimInfos.
        template <typename DimType, typename InfosTuple> struct tuple_has_dimension;

        template <typename DimType> struct tuple_has_dimension<DimType, detail::tuple<>> {
            static constexpr bool value = false;
        };

        template <typename DimType, typename FirstInfo, typename... RestInfos>
        struct tuple_has_dimension<DimType, detail::tuple<FirstInfo, RestInfos...>> {
            static constexpr bool value =
                detail::is_same<DimType, typename FirstInfo::dim_type>::value ||
                tuple_has_dimension<DimType, detail::tuple<RestInfos...>>::value;
        };

        /// @brief Find dimension info from a tuple of DimInfos.
        template <typename DimType, typename InfosTuple> struct tuple_find_dim_info_impl;

        template <typename DimType> struct tuple_find_dim_info_impl<DimType, detail::tuple<>> {
            using info = DimInfo<DimType, 0, 1>; // Dummy for error case
        };

        template <typename DimType, typename FirstInfo, typename... RestInfos>
        struct tuple_find_dim_info_impl<DimType, detail::tuple<FirstInfo, RestInfos...>> {
            static constexpr bool is_match =
                detail::is_same<DimType, typename FirstInfo::dim_type>::value;
            using info = detail::conditional_t<
                is_match, FirstInfo,
                typename tuple_find_dim_info_impl<DimType, detail::tuple<RestInfos...>>::info>;
        };

        template <typename DimType, typename InfosTuple> struct tuple_find_dim_info {
            static_assert(tuple_has_dimension<DimType, InfosTuple>::value,
                          "Dimension type not found in tensor - ensure you're using the correct "
                          "dimension type");
            using info = typename tuple_find_dim_info_impl<DimType, InfosTuple>::info;
        };
    }
}

#endif
