#ifndef SPIO_DIM_H_
#define SPIO_DIM_H_

#include "spio/macros.h"
#include "spio/mathutil.h"

namespace spio
{
    template <class DimType, int Stride>
    class Fold;

    /// @brief A base class for tensor dimensions using CRTP.
    /// Spio uses "typed tensors" which means that each tensor dimensions is a unique types.
    /// This prevents accidental mixing of different dimensions in index arithmetic
    /// expressions or subscripting operations.
    /// Normally, all dimensions classes referenced by custom tensors and indexes are
    /// generated automatically by the code generation system.
    /// @tparam Derived The derived dimension type (CRTP pattern)
    template <typename Derived>
    class Dim
    {
    public:
        DEVICE constexpr Dim(int i) : _i(i) {}

        DEVICE constexpr int get() const { return _i; }

        /// @brief Fold the dimension by a given stride.
        /// @tparam Stride the stride to fold by.
        /// @return a Fold object that is the result of folding the current dimension by the given stride.
        template <int Stride>
        DEVICE constexpr Fold<Derived, Stride> fold() const
        {
            return Fold<Derived, Stride>(static_cast<const Derived &>(*this));
        }

        /// @brief Cast the dimension to a new dimension type.
        /// @tparam NewDimType the type to cast the dimension to.
        /// @return the same dimension index value in a new dimension type.
        template <class NewDimType>
        DEVICE constexpr NewDimType cast() const
        {
            return NewDimType(_i);
        }

        /// @brief  Type-safe arithmetic operators that return the derived type.
        /// @param other the index value to add to this one.
        /// @return a new index value that is the sum of this and the other.
        DEVICE constexpr Derived operator+(Derived other) const
        {
            return Derived(_i + other._i);
        }

        DEVICE constexpr Derived operator-(Derived other) const
        {
            return Derived(_i - other._i);
        }

        DEVICE constexpr Derived operator*(int scalar) const
        {
            return Derived(_i * scalar);
        }

        DEVICE constexpr Derived operator/(int scalar) const
        {
            return Derived(_i / scalar);
        }

        DEVICE constexpr Derived operator%(Derived other) const
        {
            return Derived(_i % other._i);
        }

        /// @brief Type-safe comparison operator that prevent accidental comparison of different dimension types.
        /// @param other the dimension to compare to.
        /// @return true if this dimension is less than the other, false otherwise.
        DEVICE constexpr bool operator<(Derived other) const
        {
            return _i < other._i;
        }

        DEVICE constexpr bool operator>(Derived other) const
        {
            return _i > other._i;
        }

        DEVICE constexpr bool operator<=(Derived other) const
        {
            return _i <= other._i;
        }

        DEVICE constexpr bool operator>=(Derived other) const
        {
            return _i >= other._i;
        }

        DEVICE constexpr bool operator==(Derived other) const
        {
            return _i == other._i;
        }

        DEVICE constexpr bool operator!=(Derived other) const
        {
            return _i != other._i;
        }

    private:
        const int _i;
    };

    /// @brief  A template class that implemnts a folded dimension.
    /// This class is used to represent a dimension that has been
    /// folded by a given stride.
    ///
    /// Example:
    ///      // Create an index variable c in dimension C_Dim.
    ///      C_Dim c(32);
    ///
    ///      // Fold c by 8.
    ///      auto c8 = c.fold<8>();
    ///      assert c8 == Fold<C_Dim, 8>(4);
    ///
    ///      // Unfold c8.
    ///      assert c8.unfold() == c;
    ///
    ///      // Increment c8 by 1.
    ///      assert (c8 + 1).unfold() == c + 8;
    ///
    ///      // Refold to a different stride
    ///      assert c8.fold<16>() == Fold<C_Dim, 16>(2);
    ///
    /// @tparam DimType the type that represents the dimension to fold
    /// @tparam Stride the size of the fold.
    template <class DimType, int Stride>
    class Fold : public Dim<Fold<DimType, Stride>>
    {
    public:
        using dim_type = DimType;

        constexpr static dim_type stride = Stride;

        using Base = Dim<Fold<DimType, Stride>>;
        using Base::Dim;
        using Base::get;

        explicit DEVICE constexpr Fold(const DimType dim) : Base(dim.get() / Stride) {}

        DEVICE constexpr DimType unfold() const { return DimType(get() * Stride); }

        template <int NewStride>
        DEVICE constexpr Fold<DimType, NewStride> fold() const
        {
            if constexpr (Stride > NewStride)
            {
                constexpr int relative_stride = Stride / NewStride;
                return Fold<DimType, NewStride>(get() * relative_stride);
            }
            else
            {
                return Fold<DimType, NewStride>(unfold());
            }
        }

        template <class NewDimType>
        DEVICE constexpr auto cast() const -> Fold<NewDimType, Stride> { return Fold<NewDimType, Stride>(get()); }
    };

    template <class DimType, int Size, int Stride>
    class Module : public Dim<Module<DimType, Size, Stride>>
    {
    public:
        using dim_type = DimType;
        constexpr static dim_type stride = Stride;
        constexpr static dim_type size = Size;

        using Base = Dim<Module<DimType, Size, Stride>>;

        DEVICE constexpr Module(int i) : Base(i % Size) {}

        using Base::get;

        explicit DEVICE constexpr Module(const DimType dim)
            : Base((dim.get() / Stride) % Size)
        {
        }

        DEVICE constexpr DimType unfold() const
        {
            return DimType(get() * Stride);
        }
    };

    /// @brief A template class that implements a range of indexes in a dimension.
    /// This class is used internally by the range functions below.
    /// @tparam dim_type The type of the dimension.
    /// @tparam increment The increment value.
    template <typename dim_type, int increment = 1>
    class _Range
    {
    public:
        class Iterator
        {
        public:
            DEVICE constexpr Iterator(int i) : _i(i) {}
            DEVICE dim_type operator*() const { return _i; }
            DEVICE constexpr Iterator &operator++()
            {
                _i += increment;
                return *this;
            }
            DEVICE constexpr bool operator!=(const Iterator other) const { return _i != other._i; }

        private:
            int _i;
        };

        DEVICE constexpr _Range(dim_type end) : _start(0), _end(end.get()) {}

        DEVICE constexpr _Range(dim_type start, dim_type end) : _start(start.get()), _end(end.get()) {}

        DEVICE constexpr Iterator begin() const { return Iterator(_start); }

        DEVICE constexpr Iterator end() const { return Iterator(_end); }

    private:
        int _start;
        int _end;
    };

    /// @brief A template class that implements a reverse range of indexes in a dimension.
    /// This class is used internally by the reverse_range functions below.
    /// @tparam dim_type The type of the dimension.
    template <typename dim_type>
    class _ReverseRange
    {
    public:
        class ReverseIterator
        {
        public:
            DEVICE constexpr ReverseIterator(int i) : _i(i) {}
            DEVICE dim_type operator*() const { return dim_type(_i); }
            DEVICE constexpr ReverseIterator &operator++()
            {
                --_i;
                return *this;
            }
            DEVICE constexpr bool operator!=(const ReverseIterator other) const { return _i != other._i; }

        private:
            int _i;
        };

        // Create range from 0 to end (reverse)
        DEVICE constexpr _ReverseRange(dim_type end) : _start(end.get() - 1), _end(-1) {}

        // Create range from start to end (reverse)
        DEVICE constexpr _ReverseRange(dim_type start, dim_type end) : _start(end.get() - 1), _end(start.get() - 1) {}

        DEVICE constexpr ReverseIterator begin() const { return ReverseIterator(_start); }
        DEVICE constexpr ReverseIterator end() const { return ReverseIterator(_end); }

    private:
        int _start; // Highest value (starting point)
        int _end;   // Just below lowest value (ending point)
    };

    /// @brief Returns a range of integers from 0 to end, incrementing by increment.
    template <int increment, typename dim_type>
    DEVICE constexpr auto range_with_step(dim_type end)
    {
        return _Range<dim_type, increment>(divup(end.get(), increment) * increment);
    }

    /// @brief Returns a range of integers from start to end, incrementing by increment.
    template <int increment, typename dim_type>
    DEVICE constexpr auto range_with_step(dim_type start, dim_type end)
    {
        return _Range<dim_type, increment>(start, start + divup(end.get() - start.get(), increment) * increment);
    }

    /// @brief Returns a range of integers from 0 to end, incrementing by 1.
    template <typename dim_type>
    DEVICE constexpr auto range(dim_type end)
    {
        return _Range<dim_type>(end);
    }

    /// @brief Returns a range of integers from start to end, incrementing by 1.
    template <typename dim_type>
    DEVICE constexpr auto range(dim_type start, dim_type end)
    {
        return _Range<dim_type>(start, end);
    }

    /// @brief Returns a range of integers from end-1 down to 0, decrementing by 1.
    template <typename dim_type>
    DEVICE constexpr auto reverse_range(dim_type end)
    {
        return _ReverseRange<dim_type>(end);
    }

    /// @brief Returns a range of integers from end-1 down to start, decrementing by 1.
    template <typename dim_type>
    DEVICE constexpr auto reverse_range(dim_type start, dim_type end)
    {
        return _ReverseRange<dim_type>(start, end);
    }
}

#endif