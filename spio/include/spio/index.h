#ifndef SPIO_INDEX_H_
#define SPIO_INDEX_H_

#include "spio/macros.h"
#include "spio/index_base.h"
#include "spio/dim.h"
#include "spio/dim_info.h"

namespace spio
{
    namespace detail {
        // Simple tuple implementation
        template<typename... Ts>
        struct tuple;
        
        template<>
        struct tuple<> {};
        
        template<typename T, typename... Ts>
        struct tuple<T, Ts...> {
            T first;
            tuple<Ts...> rest;
        };
                
        // Helper for type decay to remove reference/const
        template<typename T>
        struct decay {
            using type = T;
        };
        
        template<typename T>
        struct decay<T&> {
            using type = T;
        };
        
        template<typename T>
        struct decay<const T> {
            using type = T;
        };
        
        template<typename T>
        struct decay<const T&> {
            using type = T;
        };
        
        template<typename T>
        using decay_t = typename decay<T>::type;
        
        // Helper template to compute product of sizes at compile time
        template<typename... Ts>
        struct product_sizes;
        
        template<typename T, typename... Ts>
        struct product_sizes<T, Ts...>
        {
            static constexpr int value = T::module_type::size.get() * product_sizes<Ts...>::value;
        };
        
        template<typename T>
        struct product_sizes<T>
        {
            static constexpr int value = T::module_type::size.get();
        };

        // Add a new trait to identify dummy dimensions
        template<typename DimInfo>
        struct is_dummy_dimension {
            static constexpr bool value = false;
        };
        
        // Specialization will be added via template specialization in generated code
    }

    // Adding an index_traits namespace for Index-specific traits
    namespace index_traits {
        template<typename... DimInfos>
        struct total_elements {
            static constexpr int value = detail::product_sizes<DimInfos...>::value;
        };
    }

    /// @brief Index class for mapping linear offsets to multidimensional coordinates
    /// @details This class is the inverse of Tensor - it maps a linear offset
    /// (like a thread index) back to typed dimension coordinates
    /// @tparam DimInfos The dimension information types (same as in Tensor)
    template <typename... DimInfos>
    class Index : public IndexBase
    {
    public:
        // Total number of elements (product of all dimension sizes)
        static constexpr int total_size = index_traits::total_elements<DimInfos...>::value;
        
        using IndexBase::IndexBase;
               
        /// @brief Get the typed coordinate for a specific dimension
        /// @tparam DimType The dimension type to extract
        /// @return A typed dimension value
        template <typename DimType>
        DEVICE constexpr DimType get() const {
            constexpr unsigned size = dim_traits::dimension_size<DimType, DimInfos...>::value.get();
            constexpr unsigned stride = dim_traits::dimension_stride<DimType, DimInfos...>::value.get();
            return DimType((offset() / stride) % size);
        }
        
        // Alternative method form if you prefer function syntax
        DEVICE static constexpr int size() { return total_size; }

        // Get size for a specific dimension
        template <typename DimType>
        DEVICE static constexpr DimType size()
        {
            return dim_traits::dimension_size<DimType, DimInfos...>::value;
        }

        //@brief  Apply the index to a tensor or cursor.
        //@details Applies every dimension of the index to the tensor or cursor's subscript operator.
        //@tparam TensorOrCursor the type of the tensor or cursor to apply the index to
        //@return a new tensor or cursor with the index applied
        template <typename TensorOrCursor>
        DEVICE constexpr auto apply_to(TensorOrCursor tensor) const {
            // Start the recursive application with the first dimension
            return apply_dimensions<DimInfos...>(tensor);
        }

    private:
        // Base case: no more dimensions to apply
        template <typename TensorOrCursor>
        DEVICE constexpr auto apply_dimensions(TensorOrCursor tensor) const {
            return tensor;
        }

        // Recursive case: apply the first dimension, then recurse with the rest
        template <typename FirstDimInfo, typename... RestDimInfos, typename TensorOrCursor>
        DEVICE constexpr auto apply_dimensions(TensorOrCursor tensor) const {
            // Skip dummy dimensions
            if constexpr (detail::is_dummy_dimension<FirstDimInfo>::value) {
                return apply_dimensions<RestDimInfos...>(tensor);
            } else {
                // Get this dimension's value using its type
                using CurrentDimType = typename FirstDimInfo::dim_type;
                auto dim_value = get<CurrentDimType>();
                
                // Apply it to the tensor
                auto next_tensor = tensor[dim_value];
                
                // Continue recursively with remaining dimensions
                return apply_dimensions<RestDimInfos...>(next_tensor);
            }
        }
    };
}

#endif
