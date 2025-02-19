#ifndef SPIO_FRAGMENT_INDEX_H_
#define SPIO_FRAGMENT_INDEX_H_

#include "spio/macros.h"
#include "spio/index.h"

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

    /// @brief Base class for matrix A and accumulator C that with 8x8 fragments.
    class _MMA_AC_88_F16_F32_Index : public _MMA_88_Index
    {
    public:
        // Return major-axis index divided by 8 for the given fragment index.
        DEVICE inline constexpr int _x8(int idx = 0) const { return (idx & 1); }

        // Return the minor-axis index divided by 8 for the given fragment index.
        DEVICE inline constexpr int _y8(int idx = 0) const { return (idx >> 1); }

        using Base = _MMA_88_Index;

        using Base::Base;

        // Return the major-axis index at the given fragment-index.
        DEVICE inline constexpr int _x(int idx = 0) const { return Base::_x() + (_x8(idx) << 3); }

        // Return the minor-axis index divided by 2 at the given fragment index.
        DEVICE inline constexpr int _y2(int idx = 0) const { return Base::_y2m4() + (_y8(idx) << 2); }
    };

    /// @brief Matrix A index for 8x8 fragments with float16 elements.
    class MMA_A_88_F16_Index : public _MMA_AC_88_F16_F32_Index
    {
    public:
        using Base = _MMA_AC_88_F16_F32_Index;

        using Base::Base;

        // Return the row number the given fragment index.
        DEVICE inline constexpr int i(int idx = 0) const { return Base::_x(idx); }

        // Return the column number divided by 2 at the given fragment index.
        DEVICE inline constexpr int k2(int idx = 0) const { return Base::_y2(idx); }

        // Return the column number divided by 8 at the given fragment index.
        DEVICE inline constexpr int k8(int idx = 0) const { return _y8(idx); }

        // Return the column number divided by 2 modulo 4 at the given fragment index.
        DEVICE inline constexpr int k2m4() const { return Base::_y2m4(); }
    };

    /// @brief Matrix C index for 8x8 fragments with float32 elements.
    class MMA_C_88_F32_Index : _MMA_AC_88_F16_F32_Index
    {
    public:
        using Base = _MMA_AC_88_F16_F32_Index;

        using Base::Base;

        // Return the row number the given fragment index.
        DEVICE inline constexpr int i(int idx = 0) const { return Base::_x(idx); }

        // Return the column number divided by 2 at the given fragment index.
        DEVICE inline constexpr int j2(int idx = 0) const { return Base::_y2(idx); }

        // Return the column number divided by 8 at the given fragment index.
        DEVICE inline constexpr int j8(int idx = 0) const { return Base::_y8(idx); }

        // Return the column number divided by 2 modulo 4 at the given fragment index.
        DEVICE inline constexpr int j2m4() const { return Base::_y2m4(); }
    };

    /// @brief  Matrix B index for 8x8 fragments with float16 elements.
    class MMA_B_88_F16_Index : public _MMA_88_Index
    {
        // Return the column number for the given fragment index.
        DEVICE inline constexpr int _j8(int idx = 0) const { return (idx >> 1); }

        // Return the row number divided by 8 for the given fragment index.
        DEVICE inline constexpr int _k8(int idx = 0) const { return (idx & 1); }

    public:
        using Base = _MMA_88_Index;

        using Base::Base;

        // Return the column number at the given fragment index.
        DEVICE inline constexpr int j(int idx = 0) const { return Base::_x() + (_j8(idx) << 3); }

        // Return the row number divided by 2 at the given fragment index.
        DEVICE inline constexpr int k2(int idx = 0) const { return Base::_y2m4() + (_k8(idx) << 2); }

        // Return the row number divided by 8 at the given fragment index.
        DEVICE inline constexpr int k8(int idx = 0) const { return _k8(idx); }

        // Return the row number divided by 2 modulo 4 at the given fragment index.
        DEVICE inline constexpr int k2m4() const { return Base::_y2m4(); }
    };
}

#endif
