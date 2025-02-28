#ifndef SPIO_DIM_H_
#define SPIO_DIM_H_

#include "spio/macros.h"
#include "spio/mathutil.h"

namespace spio
{
    /// @brief A base class for tensor dimensions.
    /// Users generate custom dimension subclasses for the dimensions they need.
    /// Normally, all dimensions classes referenced by custom tensors and indexes are
    /// generated automatically by the code generation system.
    class Dim
    {
    public:
        DEVICE constexpr Dim(int i) : _i(i) {}

        DEVICE constexpr int get() const { return _i; }

    protected:
        DEVICE constexpr int _add(const Dim other) const { return _i + other._i; }
        DEVICE constexpr int _sub(const Dim other) const { return _i - other._i; }
        DEVICE constexpr bool operator<(const Dim other) const { return _i < other._i; }
        DEVICE constexpr bool operator==(const Dim other) const { return _i == other._i; }
        DEVICE constexpr int _modulus(const Dim other) const { return _i % other._i; }

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
    template <class DimType, unsigned Stride>
    class Fold
    {
    public:
        using dim_type = DimType;

        constexpr static dim_type stride = Stride;

        DEVICE constexpr Fold(int fold) : _fold(fold) {}

        explicit DEVICE constexpr Fold(const DimType dim) : _fold(dim.get() / Stride) {}

        DEVICE constexpr int get() const { return _fold; }

        DEVICE constexpr DimType unfold() const { return DimType(_fold * Stride); }

        template <unsigned NewStride>
        DEVICE constexpr Fold<DimType, NewStride> fold() const { return Fold<DimType, NewStride>(_fold * Stride / NewStride); }

        template <class NewDimType>
        DEVICE constexpr auto cast() const -> Fold<NewDimType, Stride> { return Fold<NewDimType, Stride>(_fold); }

        class Iterator
        {
        public:
            DEVICE constexpr Iterator(int i) : _j(i) {}
            DEVICE constexpr Iterator operator++()
            {
                ++_j;
                return *this;
            }
            DEVICE constexpr bool operator!=(const Iterator other) const { return _j != other._j; }
            DEVICE constexpr Fold operator*() const { return Fold(_j); }

        private:
            int _j;
        };

        DEVICE constexpr Iterator begin() const { return Iterator(0); }
        DEVICE constexpr Iterator end() const { return Iterator(_fold); }

        DEVICE constexpr bool operator<(const Fold other) const { return _fold < other._fold; }

        DEVICE constexpr bool operator>(const Fold other) const { return _fold > other._fold; }

        DEVICE constexpr bool operator<=(const Fold other) const { return _fold <= other._fold; }

        DEVICE constexpr bool operator>=(const Fold other) const { return _fold >= other._fold; }

        DEVICE constexpr bool operator==(const Fold other) const { return _fold == other._fold; }

        DEVICE constexpr Fold operator+(const Fold other) const { return Fold(_fold + other._fold); }

        DEVICE constexpr Fold operator-(const Fold other) const { return Fold(_fold - other._fold); }

        DEVICE constexpr Fold operator%(const Fold other) const { return Fold(_fold % other._fold); }

    private:
        const int _fold;
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

    /// @brief Returns a range of integers from 0 to end, incrementing by increment.
    /// Adjusts the end-of-range to ensure it is a multiple of the increment.
    /// @param increment The increment value.
    /// @param end The end value.
    /// @return A range of integers.
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
}

#endif