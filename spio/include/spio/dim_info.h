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

    /// @brief Store information about a tensor dimension.
    /// @tparam DimType the dimension type
    /// @tparam Size the size of the dimension
    /// @tparam Stride the stride of the dimension
    template <typename DimType, int Size, int Stride> struct DimInfo {
        using dim_type = DimType;
        static constexpr int size = Size;
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
    }
}

#endif
