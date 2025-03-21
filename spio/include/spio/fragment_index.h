#ifndef SPIO_FRAGMENT_INDEX_H_
#define SPIO_FRAGMENT_INDEX_H_

#include "spio/macros.h"
#include "spio/index_base.h"
#include "spio/dim.h"

// Add basic type trait support if not available
namespace spio_internal {
    // Basic implementation of is_same type trait
    template<typename T, typename U>
    struct is_same {
        static constexpr bool value = false;
    };
    
    template<typename T>
    struct is_same<T, T> {
        static constexpr bool value = true;
    };
}

// Define in std namespace to match standard usage
namespace std {
    // Use our implementation if std::is_same isn't available
    #if __cplusplus < 201103L || defined(__CUDACC__)
    template<typename T, typename U>
    struct is_same : public spio_internal::is_same<T, U> {};
    
    template<typename T, typename U>
    inline constexpr bool is_same_v = is_same<T, U>::value;
    #endif
}

namespace spio
{
    /// @brief Base class for all MMA indices that use 8x8 fragments.
    class _MMA_88_Index : public IndexBase
    {
    public:
        using IndexBase::IndexBase;

        DEVICE inline constexpr int _x() const { return offset() >> 2; };
        DEVICE inline constexpr int _y2m4() const { return offset() & 3; };
    };

    /// @brief Matrix A index for 8x8 fragments with float16 elements.
    /// @tparam RowDim The dimension type for rows (i)
    /// @tparam ColDim The dimension type for columns (k)
    template <typename RowDim, typename ColDim>
    class MMA_A_88_F16_Index : public _MMA_88_Index
    {
    private:
        using Base = _MMA_88_Index;
        
        // Helper method to get row index at given fragment index
        DEVICE inline constexpr int _i(int idx = 0) const { 
            return Base::_x() + ((idx & 1) << 3); 
        }
        
        // Helper method to get column index divided by 2
        DEVICE inline constexpr int _k2(int idx = 0) const { 
            return Base::_y2m4() + ((idx >> 1) << 2); 
        }
        
        // Helper for column index divided by 8
        DEVICE inline constexpr int _k8(int idx = 0) const { 
            return (idx >> 1); 
        }

    public:
        using Base::Base;
        
        // Row dimension access - template specialization
        template <typename Dim>
        DEVICE constexpr auto get(int idx = 0) const {
            if constexpr (std::is_same_v<Dim, RowDim>) {
                return RowDim(_i(idx));
            } else if constexpr (std::is_same_v<Dim, Fold<ColDim, 2>>) {
                return Fold<ColDim, 2>(_k2(idx));
            } else if constexpr (std::is_same_v<Dim, Fold<ColDim, 8>>) {
                return Fold<ColDim, 8>(_k8(idx));
            } else if constexpr (std::is_same_v<Dim, Module<ColDim, 4, 2>>) {
                return Module<ColDim, 4, 2>(Base::_y2m4());
            } else {
                // Static assertion to provide better error message
                static_assert(
                    std::is_same_v<Dim, RowDim> || 
                    std::is_same_v<Dim, Fold<ColDim, 2>> ||
                    std::is_same_v<Dim, Fold<ColDim, 8>> ||
                    std::is_same_v<Dim, Module<ColDim, 4, 2>>,
                    "Invalid dimension type for MMA_A_88_F16_Index"
                );
                return Dim(0); // Unreachable but needed for compilation
            }
        }
        
        // Convenience methods for backward compatibility
        DEVICE inline constexpr RowDim row(int idx = 0) const { 
            return get<RowDim>(idx); 
        }
        
        DEVICE inline constexpr Fold<ColDim, 2> col2(int idx = 0) const { 
            return get<Fold<ColDim, 2>>(idx); 
        }
        
        DEVICE inline constexpr Fold<ColDim, 8> col8(int idx = 0) const { 
            return get<Fold<ColDim, 8>>(idx); 
        }
        
        DEVICE inline constexpr Module<ColDim, 4, 2> col2_mod4() const { 
            return get<Module<ColDim, 4, 2>>(); 
        }
    };

    /// @brief Matrix B index for 8x8 fragments with float16 elements.
    /// @tparam RowDim The dimension type for rows (k)
    /// @tparam ColDim The dimension type for columns (j)
    template <typename RowDim, typename ColDim>
    class MMA_B_88_F16_Index : public _MMA_88_Index
    {
    private:
        using Base = _MMA_88_Index;
        using row_dim = RowDim;
        using col_dim = ColDim;

        
        // Helper for column index
        DEVICE inline constexpr int _j(int idx = 0) const { 
            return Base::_x() + ((idx >> 1) << 3); 
        }
        
        // Helper for row index divided by 2
        DEVICE inline constexpr int _k2(int idx = 0) const { 
            return Base::_y2m4() + ((idx & 1) << 2); 
        }
        
        // Helper for row index divided by 8
        DEVICE inline constexpr int _k8(int idx = 0) const { 
            return (idx & 1); 
        }

    public:
        using Base::Base;
        
        // Template-based dimension access
        template <typename Dim>
        DEVICE constexpr auto get(int idx = 0) const {
            if constexpr (std::is_same_v<Dim, ColDim>) {
                return ColDim(_j(idx));
            } else if constexpr (std::is_same_v<Dim, Fold<RowDim, 2>>) {
                return Fold<RowDim, 2>(_k2(idx));
            } else if constexpr (std::is_same_v<Dim, Fold<RowDim, 8>>) {
                return Fold<RowDim, 8>(_k8(idx));
            } else if constexpr (std::is_same_v<Dim, Module<RowDim, 4, 2>>) {
                return Module<RowDim, 4, 2>(Base::_y2m4());
            } else {
                static_assert(
                    std::is_same_v<Dim, ColDim> || 
                    std::is_same_v<Dim, Fold<RowDim, 2>> ||
                    std::is_same_v<Dim, Fold<RowDim, 8>> ||
                    std::is_same_v<Dim, Module<RowDim, 4, 2>>,
                    "Invalid dimension type for MMA_B_88_F16_Index"
                );
                return Dim(0);
            }
        }
        
        // Convenience methods
        DEVICE inline constexpr ColDim col(int idx = 0) const { 
            return get<ColDim>(idx); 
        }
        
        DEVICE inline constexpr Fold<RowDim, 2> row2(int idx = 0) const { 
            return get<Fold<RowDim, 2>>(idx); 
        }
        
        DEVICE inline constexpr Fold<RowDim, 8> row8(int idx = 0) const { 
            return get<Fold<RowDim, 8>>(idx); 
        }
        
        DEVICE inline constexpr Module<RowDim, 4, 2> row2_mod4() const { 
            return get<Module<RowDim, 4, 2>>(); 
        }
    };

    /// @brief Matrix C index for 8x8 fragments with float32 elements.
    /// @tparam RowDim The dimension type for rows (i)
    /// @tparam ColDim The dimension type for columns (j)
    template <typename RowDim, typename ColDim>
    class MMA_C_88_F32_Index : public _MMA_88_Index
    {
    private:
        using Base = _MMA_88_Index;
        
        // Helpers ported from _MMA_AC_88_F16_F32_Index
        DEVICE inline constexpr int _x8(int idx = 0) const { return (idx & 1); }
        DEVICE inline constexpr int _y8(int idx = 0) const { return (idx >> 1); }
        DEVICE inline constexpr int _x(int idx = 0) const { return Base::_x() + (_x8(idx) << 3); }
        DEVICE inline constexpr int _y2(int idx = 0) const { return Base::_y2m4() + (_y8(idx) << 2); }

    public:
        using Base::Base;
        
        // Template-based dimension access
        template <typename Dim>
        DEVICE constexpr auto get(int idx = 0) const {
            if constexpr (std::is_same_v<Dim, RowDim>) {
                return RowDim(_x(idx));
            } else if constexpr (std::is_same_v<Dim, Fold<ColDim, 2>>) {
                return Fold<ColDim, 2>(_y2(idx));
            } else if constexpr (std::is_same_v<Dim, Fold<ColDim, 8>>) {
                return Fold<ColDim, 8>(_y8(idx));
            } else if constexpr (std::is_same_v<Dim, Module<ColDim, 4, 2>>) {
                return Module<ColDim, 4, 2>(Base::_y2m4());
            } else {
                static_assert(
                    std::is_same_v<Dim, RowDim> || 
                    std::is_same_v<Dim, Fold<ColDim, 2>> ||
                    std::is_same_v<Dim, Fold<ColDim, 8>> ||
                    std::is_same_v<Dim, Module<ColDim, 4, 2>>,
                    "Invalid dimension type for MMA_C_88_F32_Index"
                );
                return Dim(0);
            }
        }
        
        // Convenience methods
        DEVICE inline constexpr RowDim row(int idx = 0) const { 
            return get<RowDim>(idx); 
        }
        
        DEVICE inline constexpr Fold<ColDim, 2> col2(int idx = 0) const { 
            return get<Fold<ColDim, 2>>(idx); 
        }
        
        DEVICE inline constexpr Fold<ColDim, 8> col8(int idx = 0) const { 
            return get<Fold<ColDim, 8>>(idx); 
        }
        
        DEVICE inline constexpr Module<ColDim, 4, 2> col2_mod4() const { 
            return get<Module<ColDim, 4, 2>>(); 
        }
    };
}

#endif