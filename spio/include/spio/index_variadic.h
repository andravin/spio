#ifndef SPIO_INDEX_H_
#define SPIO_INDEX_H_

#include "spio/macros.h"
#include "spio/dim.h"
#include "spio/dim_info.h"

namespace spio
{
    namespace detail {
        // Helper template to compute product of sizes at compile time
        template<typename... Ts>
        struct product_sizes;
        
        template<typename T, typename... Ts>
        struct product_sizes<T, Ts...>
        {
            static constexpr unsigned value = T::module_type::size.get() * product_sizes<Ts...>::value;
        };
        
        template<typename T>
        struct product_sizes<T>
        {
            static constexpr unsigned value = T::module_type::size.get();
        };
    }

    // Adding an index_traits namespace for Index-specific traits
    namespace index_traits {
        template<typename... DimInfos>
        struct total_elements {
            static constexpr unsigned value = detail::product_sizes<DimInfos...>::value;
        };
    }

    /// @brief Index class for mapping linear offsets to multidimensional coordinates
    /// @details This class is the inverse of Tensor - it maps a linear offset
    /// (like a thread index) back to typed dimension coordinates
    /// @tparam DimInfos The dimension information types (same as in Tensor)
    template <typename... DimInfos>
    class Index
    {
    private:
        unsigned _offset;

    public:
        // Total number of elements (product of all dimension sizes)
        static constexpr unsigned total_size = index_traits::total_elements<DimInfos...>::value;
        
        /// @brief Construct an index from a linear offset
        /// @param offset The linear offset value (e.g., from threadIdx.x)
        DEVICE constexpr Index(unsigned offset = 0) : _offset(offset) {}
        
        /// @brief Get the raw linear offset
        DEVICE constexpr unsigned offset() const { return _offset; }
        
        /// @brief Get the typed coordinate for a specific dimension
        /// @tparam DimType The dimension type to extract
        /// @return A typed dimension value
        template <typename DimType>
        DEVICE constexpr DimType get() const {
            // Use the same dim_traits infrastructure as Tensor
            return dim_traits::find_dim_info<DimType, DimInfos...>::info::from_offset(_OffsetDim(_offset));
        }
        
        /// @brief Create an index from individual typed coordinates
        /// @tparam Coords Types of the dimension coordinates
        /// @param coords The typed dimension coordinates
        /// @return An Index representing the combined coordinates
        template <typename... Coords>
        DEVICE static constexpr Index from_coords(Coords... coords) {
            unsigned offset = 0;
            
            // Sum the contributions from each dimension
            // Use fold expression if using C++17
            (void)std::initializer_list<int>{
                (offset += dim_traits::find_dim_info<
                    typename std::decay<decltype(coords)>::type, 
                    DimInfos...>::info::to_offset(coords).get(), 0)...
            };
            
            return Index(offset);
        }        

        // Alternative method form if you prefer function syntax
        DEVICE static constexpr unsigned size() { return total_size; }
    };
}

#endif
