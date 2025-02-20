#ifndef SPIO_DIM_H_
#define SPIO_DIM_H_

#include "spio/macros.h"

namespace spio
{
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

}

#endif