#ifndef SPIO_FRAGMENT_LOAD_INDEX_H_
#define SPIO_FRAGMENT_LOAD_INDEX_H_

#include "spio/macros.h"
#include "spio/index.h"

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

    /// @brief Indices for a A-matrix shape M16 x K8 x float16 for use with ldmatrix.
    class MMA_A_M16_K8_F16_LoadIndex : public _MMA_A_88_F16_LoadIndex
    {
    public:
        using Base = _MMA_A_88_F16_LoadIndex;

        using Base::Base;

        DEVICE inline constexpr int i() const { return Base::_i0(); }

        DEVICE inline constexpr int k8() const { return 0; };
    };

    /// @brief Indices for a A-matrix shape M16 x K16 x float16 for use with ldmatrix.
    class MMA_A_M16_K16_F16_LoadIndex : public _MMA_A_88_F16_LoadIndex
    {
    public:
        using Base = _MMA_A_88_F16_LoadIndex;

        using Base::Base;

        DEVICE inline constexpr int i() const { return Base::_i0(); }

        DEVICE inline constexpr int k8() const { return Base::_k8(); }
    };

    /// @brief Indices for B-matrix shape N8 x K8 x float16 for use with ldmatrix.
    class MMA_B_N8_K8_F16_LoadIndex : public _MMA_B_88_F16_LoadIndex
    {
    public:
        using Base = _MMA_B_88_F16_LoadIndex;

        using Base::Base;

        DEVICE inline constexpr int j() const { return Base::_j0(); }

        DEVICE inline constexpr int k8() const { return 0; }
    };

    /// @brief Indices for B-matrix shape N8 x K16 x float16 for use with ldmatrix.
    class MMA_B_N8_K16_F16_LoadIndex : public _MMA_B_88_F16_LoadIndex
    {
    public:
        using Base = _MMA_B_88_F16_LoadIndex;

        using Base::Base;

        DEVICE inline constexpr int j() const { return Base::_j0(); }

        DEVICE inline constexpr int k8() const { return Base::_k8(); }
    };

    /// @brief Indices for B-matrix shape N16 x K16 x float16 for use with ldmatrix.
    class MMA_B_N16_K16_F16_LoadIndex : _MMA_B_88_F16_LoadIndex
    {
    public:
        using Base = _MMA_B_88_F16_LoadIndex;

        using Base::Base;

        DEVICE inline constexpr int j() const { return Base::_j0() + Base::_j8(); }

        DEVICE inline constexpr int k8() const { return Base::_k8(); }
    };
}

#endif
