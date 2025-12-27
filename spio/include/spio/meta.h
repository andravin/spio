#ifndef SPIO_META_H
#define SPIO_META_H

#include "spio/macros.h"

#if defined(__CUDACC_RTC__)
namespace std {
    using size_t = decltype(sizeof(0));
}
#else
#include <cstddef>
#endif

namespace spio {
    // Forward declaration needed for replace_base_dim specialization
    template <class DimType, int Stride> class Fold;
    template <class DimType, int Size, int Stride> class Module;

    namespace detail {

        // These must be defined first as they're used throughout
        template <bool B> struct bool_constant {
            static constexpr bool value = B;
        };

        using true_type = bool_constant<true>;
        using false_type = bool_constant<false>;

        // Minimal is_void type trait for CUDA/NVRTC compatibility
        template <typename T> struct is_void : false_type {};

        template <> struct is_void<void> : true_type {};

        template <typename T> inline constexpr bool is_void_v = is_void<T>::value;

        template <typename...> using void_t = void;

        template <typename T> T&& declval();

        template <typename T> struct dependent_false : false_type {};

        template <typename T, typename U> struct is_same : false_type {};

        template <typename T> struct is_same<T, T> : true_type {};

        template <bool Condition, typename TrueType, typename FalseType> struct conditional {
            using type = FalseType;
        };

        template <typename TrueType, typename FalseType>
        struct conditional<true, TrueType, FalseType> {
            using type = TrueType;
        };

        template <bool Condition, typename TrueType, typename FalseType>
        using conditional_t = typename conditional<Condition, TrueType, FalseType>::type;

        template <typename T> struct remove_reference {
            using type = T;
        };

        template <typename T> struct remove_reference<T&> {
            using type = T;
        };

        template <typename T> struct remove_reference<T&&> {
            using type = T;
        };

        template <typename T> using remove_reference_t = typename remove_reference<T>::type;

        template <typename T> struct remove_const {
            using type = T;
        };

        template <typename T> struct remove_const<const T> {
            using type = T;
        };

        template <typename T> using remove_const_t = typename remove_const<T>::type;

        template <typename T> struct decay {
            using type = remove_const_t<remove_reference_t<T>>;
        };

        template <typename T> using decay_t = typename decay<T>::type;

        template <typename... Ts> struct tuple;

        template <> struct tuple<> {
            constexpr tuple() = default;
        };

        template <typename T, typename... Ts> struct tuple<T, Ts...> {
            T first;
            tuple<Ts...> rest;

            constexpr tuple() = default;
            constexpr tuple(const tuple&) = default;
            constexpr tuple(tuple&&) = default;

            // Add assignment operators
            constexpr tuple& operator=(const tuple&) = default;
            constexpr tuple& operator=(tuple&&) = default;

            template <typename U, typename... Us>
            DEVICE constexpr tuple(U&& first_value, Us&&... rest_values)
                : first(static_cast<U&&>(first_value)),
                  rest(static_cast<Us&&>(rest_values)...) {}
        };

        template <typename Tuple> struct tuple_size;

        template <typename... Ts> struct tuple_size<tuple<Ts...>> {
            static constexpr std::size_t value = sizeof...(Ts);
        };

        template <std::size_t I, typename T, typename... Ts>
        DEVICE constexpr auto& get(tuple<T, Ts...>& t) {
            static_assert(I < tuple_size<tuple<T, Ts...>>::value, "tuple index out of range");
            if constexpr (I == 0) {
                return t.first;
            } else {
                return get<I - 1>(t.rest);
            }
        }

        template <std::size_t I, typename T, typename... Ts>
        DEVICE constexpr const auto& get(const tuple<T, Ts...>& t) {
            static_assert(I < tuple_size<tuple<T, Ts...>>::value, "tuple index out of range");
            if constexpr (I == 0) {
                return t.first;
            } else {
                return get<I - 1>(t.rest);
            }
        }

        template <std::size_t I, typename T, typename... Ts>
        DEVICE constexpr auto&& get(tuple<T, Ts...>&& t) {
            static_assert(I < tuple_size<tuple<T, Ts...>>::value, "tuple index out of range");
            if constexpr (I == 0) {
                return static_cast<T&&>(t.first);
            } else {
                return get<I - 1>(static_cast<tuple<Ts...>&&>(t.rest));
            }
        }

        template <std::size_t I, typename T, typename... Ts>
        DEVICE constexpr const auto&& get(const tuple<T, Ts...>&& t) {
            static_assert(I < tuple_size<tuple<T, Ts...>>::value, "tuple index out of range");
            if constexpr (I == 0) {
                return static_cast<const T&&>(t.first);
            } else {
                return get<I - 1>(static_cast<const tuple<Ts...>&&>(t.rest));
            }
        }

        template <typename Target, typename Tuple> struct tuple_contains : false_type {};

        template <typename Target, typename Head, typename... Tail>
        struct tuple_contains<Target, tuple<Head, Tail...>>
            : conditional_t<is_same<Target, Head>::value, true_type,
                            tuple_contains<Target, tuple<Tail...>>> {};

        template <typename Target, typename Tuple>
        inline constexpr bool tuple_contains_v = tuple_contains<Target, Tuple>::value;

        // Get tuple element type by index
        template <std::size_t I, typename Tuple> struct tuple_element;

        template <typename Head, typename... Tail> struct tuple_element<0, tuple<Head, Tail...>> {
            using type = Head;
        };

        template <std::size_t I, typename Head, typename... Tail>
        struct tuple_element<I, tuple<Head, Tail...>> {
            using type = typename tuple_element<I - 1, tuple<Tail...>>::type;
        };

        template <std::size_t I, typename Tuple>
        using tuple_element_t = typename tuple_element<I, Tuple>::type;

        template <typename DimType> DEVICE constexpr DimType& tuple_get_by_type(tuple<>&) {
            static_assert(dependent_false<DimType>::value, "DimType not found in tuple.");
            return *reinterpret_cast<DimType*>(nullptr);
        }

        template <typename DimType, typename Head, typename... Tail>
        DEVICE constexpr DimType& tuple_get_by_type(tuple<Head, Tail...>& t) {
            if constexpr (is_same<DimType, Head>::value) {
                return t.first;
            } else {
                return tuple_get_by_type<DimType>(t.rest);
            }
        }

        template <typename DimType>
        DEVICE constexpr const DimType& tuple_get_by_type(const tuple<>&) {
            static_assert(dependent_false<DimType>::value, "DimType not found in tuple.");
            return *reinterpret_cast<const DimType*>(nullptr);
        }

        template <typename DimType, typename Head, typename... Tail>
        DEVICE constexpr const DimType& tuple_get_by_type(const tuple<Head, Tail...>& t) {
            if constexpr (is_same<DimType, Head>::value) {
                return t.first;
            } else {
                return tuple_get_by_type<DimType>(t.rest);
            }
        }

        template <bool B, typename T = void> struct enable_if {};

        template <typename T> struct enable_if<true, T> {
            using type = T;
        };

        template <bool B, typename T = void> using enable_if_t = typename enable_if<B, T>::type;

        // tuple_cat - concatenate two tuples
        template <typename Tuple1, typename Tuple2> struct tuple_cat_impl;

        template <typename... Ts, typename... Us>
        struct tuple_cat_impl<tuple<Ts...>, tuple<Us...>> {
            using type = tuple<Ts..., Us...>;
        };

        template <typename Tuple1, typename Tuple2>
        using tuple_cat_t = typename tuple_cat_impl<Tuple1, Tuple2>::type;

        // tuple_filter_out - keep types from Tuple1 that do NOT exist in Tuple2
        template <typename Tuple1, typename Tuple2, typename Result = tuple<>>
        struct tuple_filter_out;

        template <typename Tuple2, typename Result>
        struct tuple_filter_out<tuple<>, Tuple2, Result> {
            using type = Result;
        };

        template <typename Head, typename... Tail, typename Tuple2, typename... ResultTypes>
        struct tuple_filter_out<tuple<Head, Tail...>, Tuple2, tuple<ResultTypes...>> {
            using type = typename tuple_filter_out<
                tuple<Tail...>, Tuple2,
                conditional_t<tuple_contains<Head, Tuple2>::value, tuple<ResultTypes...>,
                              tuple<ResultTypes..., Head>>>::type;
        };

        template <typename Tuple1, typename Tuple2>
        using tuple_filter_out_t = typename tuple_filter_out<Tuple1, Tuple2>::type;

        // Check if two tuples share any exact type (not just base dims)
        template <typename Tuple1, typename Tuple2> struct tuples_share_type;

        template <typename Tuple2> struct tuples_share_type<tuple<>, Tuple2> : false_type {};

        template <typename Head, typename... Tail, typename Tuple2>
        struct tuples_share_type<tuple<Head, Tail...>, Tuple2>
            : conditional_t<tuple_contains_v<Head, Tuple2>, true_type,
                            tuples_share_type<tuple<Tail...>, Tuple2>> {};

        template <typename Tuple1, typename Tuple2>
        inline constexpr bool tuples_share_type_v = tuples_share_type<Tuple1, Tuple2>::value;

        // index_sequence implementation for CUDA RTC compatibility
        template <std::size_t... Is> struct index_sequence {};

        template <std::size_t N, std::size_t... Is>
        struct make_index_sequence_impl : make_index_sequence_impl<N - 1, N - 1, Is...> {};

        template <std::size_t... Is> struct make_index_sequence_impl<0, Is...> {
            using type = index_sequence<Is...>;
        };

        template <std::size_t N>
        using make_index_sequence = typename make_index_sequence_impl<N>::type;

        // ========================================================================
        // Dimension type traits - single definitions
        // ========================================================================

        // Get the size of a dimension type
        // For base Dims and Folds: unbounded (represented as max int)
        // For Modules: actual compile-time size
        template <typename T, typename = void> struct get_dim_size {
            static constexpr int value = 0x7FFFFFFF; // Unbounded
        };

        template <typename T> struct get_dim_size<T, void_t<decltype(T::size)>> {
            static constexpr int value = T::size;
        };

        template <typename T> inline constexpr int get_dim_size_v = get_dim_size<T>::value;

        // Check if a dimension has bounded size (i.e., is a Module)
        template <typename T>
        inline constexpr bool is_bounded_v = get_dim_size<T>::value != 0x7FFFFFFF;

        // Check if two dim-like types are compatible (same base dimension)
        // Returns false if either type doesn't have dim_type
        template <typename T, typename U, typename = void> struct dims_compatible : false_type {};

        template <typename T, typename U>
        struct dims_compatible<T, U, void_t<typename T::dim_type, typename U::dim_type>>
            : bool_constant<is_same<typename T::dim_type, typename U::dim_type>::value> {};

        template <typename T, typename U>
        inline constexpr bool dims_compatible_v = dims_compatible<T, U>::value;

        // ========================================================================
        // Detection for tensor_type member
        // ========================================================================

        // Check if a type has a tensor_type member alias
        template <typename T, typename = void> struct has_tensor_type : false_type {};

        template <typename T>
        struct has_tensor_type<T, void_t<typename T::tensor_type>> : true_type {};

        template <typename T> inline constexpr bool has_tensor_type_v = has_tensor_type<T>::value;

        // ========================================================================
        // Tuple operations for dimension types
        // ========================================================================

        // Check if tuple contains a type with matching base dim_type
        template <typename BaseDimType, typename Tuple> struct tuple_contains_base_dim;

        template <typename BaseDimType>
        struct tuple_contains_base_dim<BaseDimType, tuple<>> : false_type {};

        template <typename BaseDimType, typename Head, typename... Tail>
        struct tuple_contains_base_dim<BaseDimType, tuple<Head, Tail...>>
            : conditional_t<is_same<typename Head::dim_type, BaseDimType>::value, true_type,
                            tuple_contains_base_dim<BaseDimType, tuple<Tail...>>> {};

        // Get unique base dim_types from a tuple
        template <typename Tuple, typename Result = tuple<>> struct tuple_unique_base_dims;

        template <typename Result> struct tuple_unique_base_dims<tuple<>, Result> {
            using type = Result;
        };

        template <typename Head, typename... Tail, typename... ResultTypes>
        struct tuple_unique_base_dims<tuple<Head, Tail...>, tuple<ResultTypes...>> {
            using BaseDim = typename Head::dim_type;
            using type = typename tuple_unique_base_dims<
                tuple<Tail...>,
                conditional_t<tuple_contains<BaseDim, tuple<ResultTypes...>>::value,
                              tuple<ResultTypes...>, tuple<ResultTypes..., BaseDim>>>::type;
        };

        template <typename Tuple>
        using tuple_unique_base_dims_t = typename tuple_unique_base_dims<Tuple>::type;

        // Get the first element from tuple with the given base dim type
        template <typename BaseDimType, typename Tuple> struct tuple_get_by_base_dim;

        template <typename BaseDimType> struct tuple_get_by_base_dim<BaseDimType, tuple<>> {
            template <typename T> DEVICE static constexpr auto& get(T&) {
                static_assert(dependent_false<BaseDimType>::value,
                              "BaseDimType not found in tuple.");
            }

            template <typename T> DEVICE static constexpr const auto& get(const T&) {
                static_assert(dependent_false<BaseDimType>::value,
                              "BaseDimType not found in tuple.");
            }
        };

        template <typename BaseDimType, typename Head, typename... Tail>
        struct tuple_get_by_base_dim<BaseDimType, tuple<Head, Tail...>> {
            template <typename TupleType> DEVICE static constexpr auto& get(TupleType& t) {
                if constexpr (is_same<typename Head::dim_type, BaseDimType>::value) {
                    return t.first;
                } else {
                    return tuple_get_by_base_dim<BaseDimType, tuple<Tail...>>::get(t.rest);
                }
            }

            template <typename TupleType>
            DEVICE static constexpr const auto& get(const TupleType& t) {
                if constexpr (is_same<typename Head::dim_type, BaseDimType>::value) {
                    return t.first;
                } else {
                    return tuple_get_by_base_dim<BaseDimType, tuple<Tail...>>::get(t.rest);
                }
            }
        };

        // Filter tuple to keep only types whose base dim is NOT in the other tuple's base
        // dims
        template <typename Tuple, typename OtherBaseDims, typename Result = tuple<>>
        struct tuple_exclude_base_dims;

        template <typename OtherBaseDims, typename Result>
        struct tuple_exclude_base_dims<tuple<>, OtherBaseDims, Result> {
            using type = Result;
        };

        template <typename Head, typename... Tail, typename OtherBaseDims, typename... ResultTypes>
        struct tuple_exclude_base_dims<tuple<Head, Tail...>, OtherBaseDims, tuple<ResultTypes...>> {
            using head_base = typename Head::dim_type;
            static constexpr bool should_exclude = tuple_contains<head_base, OtherBaseDims>::value;
            using type = typename tuple_exclude_base_dims<
                tuple<Tail...>, OtherBaseDims,
                conditional_t<should_exclude, tuple<ResultTypes...>,
                              tuple<ResultTypes..., Head>>>::type;
        };

        template <typename Tuple, typename OtherTuple>
        using tuple_exclude_base_dims_t =
            typename tuple_exclude_base_dims<Tuple, tuple_unique_base_dims_t<OtherTuple>>::type;

        // Filter tuple to keep only types whose base dim IS in the other tuple's base dims
        template <typename Tuple, typename OtherTuple, typename Result = tuple<>>
        struct tuple_keep_base_dims;

        template <typename OtherTuple, typename Result>
        struct tuple_keep_base_dims<tuple<>, OtherTuple, Result> {
            using type = Result;
        };

        template <typename Head, typename... Tail, typename OtherTuple, typename... ResultTypes>
        struct tuple_keep_base_dims<tuple<Head, Tail...>, OtherTuple, tuple<ResultTypes...>> {
            using head_base = typename Head::dim_type;
            static constexpr bool should_keep =
                tuple_contains_base_dim<head_base, OtherTuple>::value;
            using type = typename tuple_keep_base_dims<
                tuple<Tail...>, OtherTuple,
                conditional_t<should_keep, tuple<ResultTypes..., Head>,
                              tuple<ResultTypes...>>>::type;
        };

        template <typename Tuple, typename OtherTuple>
        using tuple_keep_base_dims_t = typename tuple_keep_base_dims<Tuple, OtherTuple>::type;

        // Prepend an element to a tuple
        template <typename T, typename Tuple> struct tuple_prepend;

        template <typename T, typename... Ts> struct tuple_prepend<T, tuple<Ts...>> {
            using type = tuple<T, Ts...>;
        };

        template <typename T, typename Tuple>
        using tuple_prepend_t = typename tuple_prepend<T, Tuple>::type;

        // Find a type in tuple by its base dimension
        template <typename BaseDim, typename Tuple> struct tuple_find_by_base_dim;

        template <typename BaseDim> struct tuple_find_by_base_dim<BaseDim, tuple<>> {
            // Should never get here if tuple_contains_base_dim was checked first
            using type = void;
        };

        template <typename BaseDim, typename First, typename... Rest>
        struct tuple_find_by_base_dim<BaseDim, tuple<First, Rest...>> {
            using first_base = typename First::dim_type;
            static constexpr bool matches = is_same<first_base, BaseDim>::value;
            using type =
                conditional_t<matches, First,
                              typename tuple_find_by_base_dim<BaseDim, tuple<Rest...>>::type>;
        };

        // ========================================================================
        // Base dimension replacement
        // ========================================================================

        // Replace the base dimension of a type while preserving fold structure
        // For plain Dim: X -> Y
        // For Fold<X, S>: Fold<X, S> -> Fold<Y, S>
        template <typename FromBaseDim, typename ToBaseDim, typename DimType>
        struct replace_base_dim {
            // Default: check if this is the base dim to replace
            using type = conditional_t<is_same<DimType, FromBaseDim>::value, ToBaseDim, DimType>;
        };

        // Specialization for Fold types
        template <typename FromBaseDim, typename ToBaseDim, typename InnerDim, int Stride>
        struct replace_base_dim<FromBaseDim, ToBaseDim, Fold<InnerDim, Stride>> {
            // Recursively replace in the inner dimension
            using new_inner = typename replace_base_dim<FromBaseDim, ToBaseDim, InnerDim>::type;
            using type = Fold<new_inner, Stride>;
        };

        // Specialization for Module types
        template <typename FromBaseDim, typename ToBaseDim, typename InnerDim, int Size, int Stride>
        struct replace_base_dim<FromBaseDim, ToBaseDim, Module<InnerDim, Size, Stride>> {
            using new_inner = typename replace_base_dim<FromBaseDim, ToBaseDim, InnerDim>::type;
            using type = Module<new_inner, Size, Stride>;
        };

        template <typename FromBaseDim, typename ToBaseDim, typename DimType>
        using replace_base_dim_t = typename replace_base_dim<FromBaseDim, ToBaseDim, DimType>::type;

        // ========================================================================
        // Dim-like detection
        // ========================================================================

        // Trait to detect if a type is dim-like (has a get() method returning int)
        template <typename T, typename = void> struct is_dim_like : false_type {};

        template <typename T>
        struct is_dim_like<T, void_t<decltype(declval<T>().get())>> : true_type {};

        template <typename T> inline constexpr bool is_dim_like_v = is_dim_like<T>::value;

        // ========================================================================
        // Helper to filter DimInfos by base dimension types
        // ========================================================================

        // Filter DimInfos keeping only those whose base dim is in BaseDims tuple
        template <typename DimInfosTuple, typename BaseDimsTuple> struct keep_dim_infos_by_base;

        // Base case: no more DimInfos to process
        template <typename BaseDimsTuple> struct keep_dim_infos_by_base<tuple<>, BaseDimsTuple> {
            using type = tuple<>;
        };

        // Recursive case: check if first DimInfo's base dim is in BaseDims
        template <typename FirstInfo, typename... RestInfos, typename BaseDimsTuple>
        struct keep_dim_infos_by_base<tuple<FirstInfo, RestInfos...>, BaseDimsTuple> {
            using first_base_dim = typename FirstInfo::dim_type::dim_type;
            using rest_result =
                typename keep_dim_infos_by_base<tuple<RestInfos...>, BaseDimsTuple>::type;

            using type = conditional_t<tuple_contains_v<first_base_dim, BaseDimsTuple>,
                                       tuple_prepend_t<FirstInfo, rest_result>, rest_result>;
        };

        template <typename DimInfosTuple, typename BaseDimsTuple>
        using keep_dim_infos_by_base_t =
            typename keep_dim_infos_by_base<DimInfosTuple, BaseDimsTuple>::type;

        // ========================================================================
        // Find coarsest DimInfo per base dimension (largest stride)
        // ========================================================================

        // Compare two DimInfos with same base dim, keep the one with larger stride
        template <typename InfoA, typename InfoB> struct coarser_dim_info {
            static constexpr int stride_a = InfoA::dim_type::stride;
            static constexpr int stride_b = InfoB::dim_type::stride;
            using type = conditional_t<(stride_a >= stride_b), InfoA, InfoB>;
        };

        // Find coarsest info for a given base dim from a tuple of DimInfos
        template <typename BaseDim, typename InfosTuple> struct find_coarsest_info_for_base;

        template <typename BaseDim, typename FirstInfo>
        struct find_coarsest_info_for_base<BaseDim, tuple<FirstInfo>> {
            using type = FirstInfo;
        };

        template <typename BaseDim, typename FirstInfo, typename SecondInfo, typename... RestInfos>
        struct find_coarsest_info_for_base<BaseDim, tuple<FirstInfo, SecondInfo, RestInfos...>> {
            using coarser = typename coarser_dim_info<FirstInfo, SecondInfo>::type;
            using type =
                typename find_coarsest_info_for_base<BaseDim, tuple<coarser, RestInfos...>>::type;
        };

        // Build tuple of coarsest DimInfos, one per unique base dimension
        template <typename BaseDimsTuple, typename AllInfosTuple> struct build_coarsest_infos;

        template <typename AllInfosTuple> struct build_coarsest_infos<tuple<>, AllInfosTuple> {
            using type = tuple<>;
        };

        template <typename FirstBase, typename... RestBases, typename AllInfosTuple>
        struct build_coarsest_infos<tuple<FirstBase, RestBases...>, AllInfosTuple> {
            // Filter to get only infos with this base dim
            using matching_infos = keep_dim_infos_by_base_t<AllInfosTuple, tuple<FirstBase>>;
            // Find coarsest among matching
            using coarsest = typename find_coarsest_info_for_base<FirstBase, matching_infos>::type;
            // Recurse for rest
            using rest = typename build_coarsest_infos<tuple<RestBases...>, AllInfosTuple>::type;
            using type = tuple_prepend_t<coarsest, rest>;
        };

        // Get unique base dimensions from DimInfos
        template <typename InfosTuple> struct dim_infos_unique_bases;

        template <typename... Infos> struct dim_infos_unique_bases<tuple<Infos...>> {
            using type = tuple_unique_base_dims_t<tuple<typename Infos::dim_type...>>;
        };

        // Main entry point: get coarsest DimInfo per base dimension
        template <typename InfosTuple> struct coarsest_dim_infos {
            using unique_bases = typename dim_infos_unique_bases<InfosTuple>::type;
            using type = typename build_coarsest_infos<unique_bases, InfosTuple>::type;
        };

        template <typename InfosTuple>
        using coarsest_dim_infos_t = typename coarsest_dim_infos<InfosTuple>::type;

    } // namespace detail

} // namespace spio

#endif
