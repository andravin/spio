#ifndef SPIO_DIM_INFO_H
#define SPIO_DIM_INFO_H

#include "spio/dim.h"
// Remove the standard header that's causing problems
// #include <type_traits>

namespace spio
{
    // Add our own implementations of the required type traits
    namespace detail
    {
        // Add true_type and false_type
        struct true_type {
            static constexpr bool value = true;
        };
        
        struct false_type {
            static constexpr bool value = false;
        };
        
        // Implementation of is_same
        template<typename T, typename U>
        struct is_same {
            static constexpr bool value = false;
        };
        
        template<typename T>
        struct is_same<T, T> {
            static constexpr bool value = true;
        };
        
        // Implementation of conditional_t
        template<bool Condition, typename TrueType, typename FalseType>
        struct conditional {
            using type = FalseType;
        };
        
        template<typename TrueType, typename FalseType>
        struct conditional<true, TrueType, FalseType> {
            using type = TrueType;
        };
        
        template<bool Condition, typename TrueType, typename FalseType>
        using conditional_t = typename conditional<Condition, TrueType, FalseType>::type;
    }

   /// @brief The private dimension type for linear offsets.
    class _OffsetDim : public Dim<_OffsetDim>
    {
    public:
        using Dim<_OffsetDim>::Dim;
    };

    /// @brief Store information about a tensor dimension.
    /// @tparam DimType the dimension type
    /// @tparam Size the size of the dimension
    /// @tparam Stride the stride of the dimension
    template <typename DimType, int Size, int Stride>
    struct DimInfo
    {
        using dim_type = DimType;

        /// @brief  How this dimension folds the tensor's linear offset dimension.
        using module_type = Module<_OffsetDim, Size, Stride>;
    };

    namespace detail
    {
        /// @brief Check if a dimension exists in the tensor.
        /// @tparam DimType the dimension type to check
        /// @tparam DimInfos the dimension infos
        template <typename DimType, typename... DimInfos>
        struct has_dim;

        template <typename DimType, typename FirstDimInfo, typename... RestDimInfos>
        struct has_dim<DimType, FirstDimInfo, RestDimInfos...>
        {
            static constexpr bool value =
                detail::is_same<DimType, typename FirstDimInfo::dim_type>::value ||
                has_dim<DimType, RestDimInfos...>::value;
        };

        template <typename DimType>
        struct has_dim<DimType>
        {
            static constexpr bool value = false;
        };

        template <typename DimType, typename... DimInfos>
        struct find_dim_info_impl;

        template <typename DimType, typename FirstDimInfo, typename... RestDimInfos>
        struct find_dim_info_impl<DimType, FirstDimInfo, RestDimInfos...>
        {
            static constexpr bool is_match = detail::is_same<DimType, typename FirstDimInfo::dim_type>::value;
            using info = detail::conditional_t<
                is_match,
                FirstDimInfo,
                typename find_dim_info_impl<DimType, RestDimInfos...>::info>;
        };

        /// @brief Base case with dummy DimInfo instantiation for error handling.
        template <typename DimType>
        struct find_dim_info_impl<DimType>
        {
            using info = DimInfo<DimType, 0, 1>;
        };        
    }

    // Type traits for dim_info operations
    namespace dim_traits
    {
        /// @brief Find dimension info for a given dimension type.
        /// @tparam DimType the dimension type to find
        /// @tparam ...DimInfos the dimension infos
        template <typename DimType, typename... DimInfos>
        struct find_dim_info
        {
            // First check if dimension exists and show a clear error message if it doesn't.
            static_assert(detail::has_dim<DimType, DimInfos...>::value,
                          "Dimension type not found in tensor - ensure you're using the correct dimension type");

            // Then find the dimension info.
            using impl = detail::find_dim_info_impl<DimType, DimInfos...>;
            using info = typename impl::info;
        };

        /// @brief Check if a dimension exists in the tensor (public interface).
        /// @tparam DimType the dimension type to check
        /// @tparam DimInfos the dimension infos
        template <typename DimType, typename... DimInfos>
        struct has_dimension
        {
            static constexpr bool value = detail::has_dim<DimType, DimInfos...>::value;
        };

        /// @brief Get the size of a specific dimension.
        /// @tparam DimType the dimension type
        /// @tparam DimInfos the dimension infos
        template <typename DimType, typename... DimInfos>
        struct dimension_size
        {
            static constexpr DimType value = find_dim_info<DimType, DimInfos...>::info::module_type::size.get();
        };

        template <typename DimType, typename... DimInfos>
        struct dimension_stride
        {
            static constexpr DimType value = find_dim_info<DimType, DimInfos...>::info::module_type::stride.get();
        };
    }
}


#endif
