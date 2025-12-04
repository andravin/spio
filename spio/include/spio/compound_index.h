#ifndef SPIO_COMPOUND_INDEX_H_
#define SPIO_COMPOUND_INDEX_H_

#include "spio/macros.h"
#include "spio/compound_index_base.h"
#include "spio/dim.h"
#include "spio/dim_info.h"
#include "spio/meta.h"

namespace spio {

    // Forward declaration of Coordinates
    template <typename... Dims> struct Coordinates;

    /// @brief Compound index for mapping a linear offset to multidimensional coordinates
    /// @details This class is the inverse of Tensor - it maps a linear offset
    /// (like a thread index) back to typed dimension coordinates.
    /// @tparam DimInfos The dimension information types (same as in Tensor)
    template <typename... DimInfos> class CompoundIndex : public CompoundIndexBase {
    public:
        // Total number of elements (product of all dimension sizes)
        static constexpr int total_size = detail::product_sizes<DimInfos...>::value;

        using CompoundIndexBase::CompoundIndexBase;

        /// @brief Cast dimensions by replacing all occurrences of a base dimension.
        /// @details Replaces all dimensions with base type FromDim to use base type ToDim,
        /// preserving any fold structure. For example, if the index has Fold<X, 16> and Fold<X, 8>,
        /// casting from X to Y produces Fold<Y, 16> and Fold<Y, 8>.
        /// @tparam FromDim the base dimension to replace
        /// @tparam ToDim the base dimension to replace with
        template <typename FromDim, typename ToDim> DEVICE constexpr auto cast() const {
            using NewIndex = CompoundIndex<
                typename dim_traits::replace_dimension<FromDim, ToDim, DimInfos>::type...>;
            return NewIndex(offset());
        }

        /// @brief Get the typed coordinate for a specific dimension
        /// @tparam DimType The dimension type to extract
        /// @return A typed dimension value
        template <typename DimType> DEVICE constexpr auto get() const {
            constexpr unsigned size = dim_traits::dimension_size<DimType, DimInfos...>::value.get();
            constexpr unsigned stride =
                dim_traits::dimension_stride<DimType, DimInfos...>::value.get();
            auto value = (offset() / stride) % size;
            using dim_base_type = detail::get_base_dim_type_t<DimType>;
            constexpr int dim_stride = detail::get_dim_stride<DimType>::value;
            return Module<dim_base_type, size, dim_stride>(value);
        }

        // Alternative method form if you prefer function syntax
        DEVICE static constexpr int size() {
            return total_size;
        }

        // Get size for a specific dimension
        template <typename DimType> DEVICE static constexpr DimType size() {
            return dim_traits::dimension_size<DimType, DimInfos...>::value;
        }

        /// @brief Get the number of dimensions in this index.
        DEVICE static constexpr int num_dims() {
            return sizeof...(DimInfos);
        }

        /// @brief Convert this CompoundIndex to a Coordinates object.
        /// Returns Coordinates with Module types to preserve size/stride info for optimization.
        DEVICE constexpr auto coordinates() const {
            // get<DimType>() returns Module<base_dim, size, stride>
            // We need to deduce the return type from what get() returns
            return Coordinates<decltype(get<typename DimInfos::dim_type>())...>(
                get<typename DimInfos::dim_type>()...);
        }
    };
}

#endif
