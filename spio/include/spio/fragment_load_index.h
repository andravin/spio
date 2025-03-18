#ifndef SPIO_FRAGMENT_LOAD_INDEX_H_
#define SPIO_FRAGMENT_LOAD_INDEX_H_

#include "spio/macros.h"
#include "spio/index_base.h"
#include "spio/dim.h"

/// @file
/// @brief Index classes for matrix multiply-accumulate (MMA) fragments.
///
/// The ldmatrix.xn instructions (for n in {1, 2, 4}) load n float16 matrix fragments
/// of size 8 x 8. The addresses for the rows (m) or columns (n) of each matrix fragment
/// are given by groups of 8 consecutive threads. Fragment 0 is pointed to by threads 0-7,
/// fragment 1 by threads 8-15, fragment 2 by threads 16-23, and fragment 3 by threads 24-31.
///
/// When loading multiple fragments for use with the matrix multiply-accumulate instruction,
/// we observe the fragment order that it requires. In the following tables, each
/// cell represents a matrix fragment, and the number in the cell is the fragment number
/// in the ldmatrix instruction.
///
/// Matrix A fragments:
///
///          k
///        0   8
///      +---+---+
///    0 | 0 | 2 |
/// m    +-------+
///    8 | 1 | 3 |
///      +-------+
///
/// Matrix B fragments:
///
///          k
///        0   8
///      +---+---+
///    0 | 0 | 1 |
/// n    +-------+
///    8 | 2 | 3 |
///      +-------+
///
/// The *_LoadIndex classes define the offset (i.e. lane) to row and column
/// index mapping for loading A or B matrices with the ldmatrix instruction.
/// There is a separate class for each A and B matrix size that the MMA instructions
/// support.
namespace spio
{
    /// @brief Base class for A-matrix load-index using 8x8 fragments.
    class _MMA_A_88_F16_LoadIndex : public IndexBase
    {
    public:
        using IndexBase::IndexBase;

    protected:
        DEVICE inline constexpr int _i0() const { return offset() & 15; };

        DEVICE inline constexpr int _k8() const { return offset() >> 4; };
    };

    /// @brief Base class for B-matrix load-index using 8x8 fragments.
    class _MMA_B_88_F16_LoadIndex : public IndexBase
    {
    public:
        using IndexBase::IndexBase;

    protected:
        DEVICE inline constexpr int _j0() const { return offset() & 7; }

        DEVICE inline constexpr int _j8() const { return (offset() & 16) >> 1; }

        DEVICE inline constexpr int _k8() const { return (offset() >> 3) & 1; }
    };

    /// @brief Indices for A-matrix shape M16 x K8 x float16 for use with ldmatrix.
    /// @tparam RowDim The dimension type for rows (i)
    /// @tparam ColDim The dimension type for columns (k)
    template <typename RowDim, typename ColDim>
    class MMA_A_M16_K8_F16_LoadIndex : public _MMA_A_88_F16_LoadIndex
    {
    private:
        using Base = _MMA_A_88_F16_LoadIndex;

    public:
        using Base::Base;

        /// @brief Get dimension value by type
        /// @tparam Dim The dimension type to retrieve
        /// @return The dimension value with the proper type
        template <typename Dim>
        DEVICE constexpr auto get() const {
            if constexpr (std::is_same_v<Dim, RowDim>) {
                return RowDim(Base::_i0());
            } else if constexpr (std::is_same_v<Dim, Fold<ColDim, 8>>) {
                return Fold<ColDim, 8>(0);  // Always 0 for K8
            } else {
                static_assert(
                    std::is_same_v<Dim, RowDim> || 
                    std::is_same_v<Dim, Fold<ColDim, 8>>,
                    "Invalid dimension type for MMA_A_M16_K8_F16_LoadIndex"
                );
                return Dim(0);
            }
        }
        
        // Convenience methods for backward compatibility
        DEVICE inline constexpr RowDim row() const { 
            return get<RowDim>(); 
        }
        
        DEVICE inline constexpr Fold<ColDim, 8> col8() const { 
            return get<Fold<ColDim, 8>>(); 
        }
    };

    /// @brief Indices for A-matrix shape M16 x K16 x float16 for use with ldmatrix.
    /// @tparam RowDim The dimension type for rows (i)
    /// @tparam ColDim The dimension type for columns (k)
    template <typename RowDim, typename ColDim>
    class MMA_A_M16_K16_F16_LoadIndex : public _MMA_A_88_F16_LoadIndex
    {
    private:
        using Base = _MMA_A_88_F16_LoadIndex;

    public:
        using Base::Base;

        /// @brief Get dimension value by type
        /// @tparam Dim The dimension type to retrieve
        /// @return The dimension value with the proper type
        template <typename Dim>
        DEVICE constexpr auto get() const {
            if constexpr (std::is_same_v<Dim, RowDim>) {
                return RowDim(Base::_i0());
            } else if constexpr (std::is_same_v<Dim, Fold<ColDim, 8>>) {
                return Fold<ColDim, 8>(Base::_k8());
            } else {
                static_assert(
                    std::is_same_v<Dim, RowDim> || 
                    std::is_same_v<Dim, Fold<ColDim, 8>>,
                    "Invalid dimension type for MMA_A_M16_K16_F16_LoadIndex"
                );
                return Dim(0);
            }
        }
        
        // Convenience methods for backward compatibility
        DEVICE inline constexpr RowDim row() const { 
            return get<RowDim>(); 
        }
        
        DEVICE inline constexpr Fold<ColDim, 8> col8() const { 
            return get<Fold<ColDim, 8>>(); 
        }
    };

    /// @brief Indices for B-matrix shape N8 x K8 x float16 for use with ldmatrix.
    /// @tparam RowDim The dimension type for rows (k)
    /// @tparam ColDim The dimension type for columns (j)
    template <typename RowDim, typename ColDim>
    class MMA_B_N8_K8_F16_LoadIndex : public _MMA_B_88_F16_LoadIndex
    {
    private:
        using Base = _MMA_B_88_F16_LoadIndex;

    public:
        using Base::Base;

        /// @brief Get dimension value by type
        /// @tparam Dim The dimension type to retrieve
        /// @return The dimension value with the proper type
        template <typename Dim>
        DEVICE constexpr auto get() const {
            if constexpr (std::is_same_v<Dim, ColDim>) {
                return ColDim(Base::_j0());
            } else if constexpr (std::is_same_v<Dim, Fold<RowDim, 8>>) {
                return Fold<RowDim, 8>(0);  // Always 0 for K8
            } else {
                static_assert(
                    std::is_same_v<Dim, ColDim> || 
                    std::is_same_v<Dim, Fold<RowDim, 8>>,
                    "Invalid dimension type for MMA_B_N8_K8_F16_LoadIndex"
                );
                return Dim(0);
            }
        }
        
        // Convenience methods for backward compatibility
        DEVICE inline constexpr ColDim col() const { 
            return get<ColDim>(); 
        }
        
        DEVICE inline constexpr Fold<RowDim, 8> row8() const { 
            return get<Fold<RowDim, 8>>(); 
        }
    };

    /// @brief Indices for B-matrix shape N8 x K16 x float16 for use with ldmatrix.
    /// @tparam RowDim The dimension type for rows (k)
    /// @tparam ColDim The dimension type for columns (j)
    template <typename RowDim, typename ColDim>
    class MMA_B_N8_K16_F16_LoadIndex : public _MMA_B_88_F16_LoadIndex
    {
    private:
        using Base = _MMA_B_88_F16_LoadIndex;

    public:
        using Base::Base;

        /// @brief Get dimension value by type
        /// @tparam Dim The dimension type to retrieve
        /// @return The dimension value with the proper type
        template <typename Dim>
        DEVICE constexpr auto get() const {
            if constexpr (std::is_same_v<Dim, ColDim>) {
                return ColDim(Base::_j0());
            } else if constexpr (std::is_same_v<Dim, Fold<RowDim, 8>>) {
                return Fold<RowDim, 8>(Base::_k8());
            } else {
                static_assert(
                    std::is_same_v<Dim, ColDim> || 
                    std::is_same_v<Dim, Fold<RowDim, 8>>,
                    "Invalid dimension type for MMA_B_N8_K16_F16_LoadIndex"
                );
                return Dim(0);
            }
        }
        
        // Convenience methods for backward compatibility
        DEVICE inline constexpr ColDim col() const { 
            return get<ColDim>(); 
        }
        
        DEVICE inline constexpr Fold<RowDim, 8> row8() const { 
            return get<Fold<RowDim, 8>>(); 
        }
    };

    /// @brief Indices for B-matrix shape N16 x K16 x float16 for use with ldmatrix.
    /// @tparam RowDim The dimension type for rows (k)
    /// @tparam ColDim The dimension type for columns (j)
    template <typename RowDim, typename ColDim>
    class MMA_B_N16_K16_F16_LoadIndex : public _MMA_B_88_F16_LoadIndex
    {
    private:
        using Base = _MMA_B_88_F16_LoadIndex;

    public:
        using Base::Base;

        /// @brief Get dimension value by type
        /// @tparam Dim The dimension type to retrieve
        /// @return The dimension value with the proper type
        template <typename Dim>
        DEVICE constexpr auto get() const {
            if constexpr (std::is_same_v<Dim, ColDim>) {
                return ColDim(Base::_j0() + Base::_j8());
            } else if constexpr (std::is_same_v<Dim, Fold<RowDim, 8>>) {
                return Fold<RowDim, 8>(Base::_k8());
            } else {
                static_assert(
                    std::is_same_v<Dim, ColDim> || 
                    std::is_same_v<Dim, Fold<RowDim, 8>>,
                    "Invalid dimension type for MMA_B_N16_K16_F16_LoadIndex"
                );
                return Dim(0);
            }
        }
        
        // Convenience methods for backward compatibility
        DEVICE inline constexpr ColDim col() const { 
            return get<ColDim>(); 
        }
        
        DEVICE inline constexpr Fold<RowDim, 8> row8() const { 
            return get<Fold<RowDim, 8>>(); 
        }
    };
}

#endif
