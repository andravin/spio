#ifndef SPIO_DIM_H_
#define SPIO_DIM_H_

#include "spio/macros.h"
#include "spio/mathutil.h"
#include "spio/meta.h"

namespace spio {
    template <class DimType, int Stride> class Fold;
    template <class DimType, int Size, int Stride> class Module;

    /// @brief A base class for tensor dimensions using CRTP.
    /// Spio uses "typed tensors" which means that each tensor dimensions is a unique types.
    /// This prevents accidental mixing of different dimensions in index arithmetic
    /// expressions or subscripting operations.
    /// Normally, all dimensions classes referenced by custom tensors and indexes are
    /// generated automatically by the code generation system.
    /// @tparam Derived The derived dimension type (CRTP pattern)
    template <typename Derived> class Dim {
    public:
        DEVICE constexpr Dim() : _i(0) {}

        DEVICE constexpr Dim(const Dim& other) : _i(other._i) {}

        template <int Stride>
        DEVICE constexpr Dim(const Fold<Derived, Stride> folded_dim)
            : _i(folded_dim.unfold().get()) {}

        // TODO: make explicit to prevent accidental conversion.
        // But doing so will break a lot of code.
        DEVICE constexpr Dim(int i) : _i(i) {}

        DEVICE constexpr int get() const {
            return _i;
        }

        /// @brief Fold the dimension by a given stride.
        /// @tparam Stride the stride to fold by.
        /// @return a Fold object that is the result of folding the current dimension by the given
        /// stride.
        template <int Stride> DEVICE constexpr Fold<Derived, Stride> fold() const {
            return Fold<Derived, Stride>(static_cast<const Derived&>(*this));
        }

        /// @brief Cast the dimension to a new dimension type.
        /// @tparam NewDimType the type to cast the dimension to.
        /// @return the same dimension index value in a new dimension type.
        template <class NewDimType> DEVICE constexpr NewDimType cast() const {
            return NewDimType(_i);
        }

        // Same-type arithmetic (Derived op Derived)
        DEVICE constexpr Derived operator+(Derived other) const {
            return Derived(_i + other._i);
        }

        DEVICE constexpr Derived operator-(Derived other) const {
            return Derived(_i - other._i);
        }

        DEVICE constexpr Derived operator*(int scalar) const {
            return Derived(_i * scalar);
        }

        DEVICE constexpr Derived operator/(int scalar) const {
            return Derived(_i / scalar);
        }

        DEVICE constexpr Derived operator%(Derived other) const {
            return Derived(_i % other._i);
        }

        // Same-type comparisons (Derived op Derived)
        DEVICE constexpr bool operator<(Derived other) const {
            return _i < other._i;
        }

        DEVICE constexpr bool operator>(Derived other) const {
            return _i > other._i;
        }

        DEVICE constexpr bool operator<=(Derived other) const {
            return _i <= other._i;
        }

        DEVICE constexpr bool operator>=(Derived other) const {
            return _i >= other._i;
        }

        DEVICE constexpr bool operator==(Derived other) const {
            return _i == other._i;
        }

        DEVICE constexpr bool operator!=(Derived other) const {
            return _i != other._i;
        }

    private:
        const int _i;
    };

    /// @brief A folded dimension with stride.
    template <class DimType, int Stride> class Fold : public Dim<Fold<DimType, Stride>> {
        static_assert(Stride > 0, "Fold stride must be positive");

    public:
        using dim_type = DimType;
        constexpr static dim_type stride = Stride;

        using Base = Dim<Fold<DimType, Stride>>;
        using Base::Dim;
        using Base::get;

        explicit DEVICE constexpr Fold(const DimType dim) : Base(dim.get() / Stride) {}

        template <int OtherStride>
        explicit DEVICE constexpr Fold(const Fold<DimType, OtherStride> other)
            : Base(other.template fold<Stride>().get()) {}

        DEVICE constexpr DimType unfold() const {
            return DimType(get() * Stride);
        }

        template <int NewStride> DEVICE constexpr Fold<DimType, NewStride> fold() const {
            if constexpr (Stride > NewStride) {
                constexpr int relative_stride = Stride / NewStride;
                return Fold<DimType, NewStride>(get() * relative_stride);
            } else {
                return Fold<DimType, NewStride>(unfold());
            }
        }

        template <class NewDimType> DEVICE constexpr auto cast() const -> Fold<NewDimType, Stride> {
            return Fold<NewDimType, Stride>(get());
        }
    };

    /// @brief A dimension with both size and stride (bounded).
    template <class DimType, int Size, int Stride>
    class Module : public Dim<Module<DimType, Size, Stride>> {
        static_assert(Size > 0, "Module size must be positive");
        static_assert(Stride > 0, "Module stride must be positive");

    public:
        using dim_type = DimType;
        constexpr static dim_type stride = Stride;
        constexpr static dim_type size = Size;

        using Base = Dim<Module<DimType, Size, Stride>>;

        DEVICE constexpr Module() : Base(0) {}

        DEVICE constexpr Module(int i) : Base(i % Size) {}

        using Base::get;

        explicit DEVICE constexpr Module(const DimType dim) : Base((dim.get() / Stride) % Size) {}

        template <int NewStride> DEVICE constexpr Fold<DimType, NewStride> fold() const {
            if constexpr (Stride > NewStride) {
                constexpr int relative_stride = Stride / NewStride;
                return Fold<DimType, NewStride>(get() * relative_stride);
            } else {
                return Fold<DimType, NewStride>(unfold());
            }
        }

        DEVICE constexpr auto to_fold() const {
            return fold<Stride>();
        }

        DEVICE constexpr DimType unfold() const {
            return DimType(get() * Stride);
        }
    };

    // ========================================================================
    // Cross-type operators (Dim+Fold, Fold+Module, etc.)
    // Uses ADL to find these when operands are in namespace spio
    // ========================================================================

    namespace detail {
        // Convert any dim-like type to its base value
        template <typename T> DEVICE constexpr int to_base_value(T d) {
            if constexpr (has_dim_type_v<T>) {
                return d.get() * get_dim_stride_v<T>;
            } else {
                return d.get();
            }
        }

        // Result type: use the finer (smaller) stride
        template <typename T, typename U> struct arithmetic_result_type {
            using base_type = get_base_dim_type_t<T>;
            static constexpr int t_stride = get_dim_stride_v<T>;
            static constexpr int u_stride = get_dim_stride_v<U>;
            static constexpr int result_stride = (t_stride < u_stride) ? t_stride : u_stride;
            using type =
                conditional_t<result_stride == 1, base_type, Fold<base_type, result_stride>>;
        };

        template <typename T, typename U>
        using arithmetic_result_t = typename arithmetic_result_type<T, U>::type;

        // Check if this is a cross-type operation (not same type, but compatible base)
        template <typename T, typename U>
        inline constexpr bool is_cross_type_op_v = dims_compatible_v<T, U> && !is_same<T, U>::value;
    }

    // Cross-type addition
    template <typename T, typename U,
              detail::enable_if_t<detail::is_cross_type_op_v<T, U>, int> = 0>
    DEVICE constexpr auto operator+(T lhs, U rhs) {
        using result_type = detail::arithmetic_result_t<T, U>;
        constexpr int result_stride = detail::get_dim_stride_v<result_type>;
        int base_result = detail::to_base_value(lhs) + detail::to_base_value(rhs);
        return result_type(base_result / result_stride);
    }

    // Cross-type subtraction
    template <typename T, typename U,
              detail::enable_if_t<detail::is_cross_type_op_v<T, U>, int> = 0>
    DEVICE constexpr auto operator-(T lhs, U rhs) {
        using result_type = detail::arithmetic_result_t<T, U>;
        constexpr int result_stride = detail::get_dim_stride_v<result_type>;
        int base_result = detail::to_base_value(lhs) - detail::to_base_value(rhs);
        return result_type(base_result / result_stride);
    }

    // Cross-type comparisons
    template <typename T, typename U,
              detail::enable_if_t<detail::is_cross_type_op_v<T, U>, int> = 0>
    DEVICE constexpr bool operator<(T lhs, U rhs) {
        return detail::to_base_value(lhs) < detail::to_base_value(rhs);
    }

    template <typename T, typename U,
              detail::enable_if_t<detail::is_cross_type_op_v<T, U>, int> = 0>
    DEVICE constexpr bool operator<=(T lhs, U rhs) {
        return detail::to_base_value(lhs) <= detail::to_base_value(rhs);
    }

    template <typename T, typename U,
              detail::enable_if_t<detail::is_cross_type_op_v<T, U>, int> = 0>
    DEVICE constexpr bool operator>(T lhs, U rhs) {
        return detail::to_base_value(lhs) > detail::to_base_value(rhs);
    }

    template <typename T, typename U,
              detail::enable_if_t<detail::is_cross_type_op_v<T, U>, int> = 0>
    DEVICE constexpr bool operator>=(T lhs, U rhs) {
        return detail::to_base_value(lhs) >= detail::to_base_value(rhs);
    }

    template <typename T, typename U,
              detail::enable_if_t<detail::is_cross_type_op_v<T, U>, int> = 0>
    DEVICE constexpr bool operator==(T lhs, U rhs) {
        return detail::to_base_value(lhs) == detail::to_base_value(rhs);
    }

    template <typename T, typename U,
              detail::enable_if_t<detail::is_cross_type_op_v<T, U>, int> = 0>
    DEVICE constexpr bool operator!=(T lhs, U rhs) {
        return detail::to_base_value(lhs) != detail::to_base_value(rhs);
    }

    // ========================================================================
    // Range utilities
    // ========================================================================

    template <typename dim_type, int increment = 1> class _Range {
    public:
        class Iterator {
        public:
            DEVICE constexpr Iterator(int i) : _i(i) {}

            DEVICE constexpr dim_type operator*() const {
                return dim_type(_i);
            }

            DEVICE constexpr Iterator& operator++() {
                _i += increment;
                return *this;
            }

            DEVICE constexpr bool operator!=(const Iterator other) const {
                return _i != other._i;
            }

        private:
            int _i;
        };

        DEVICE constexpr _Range(dim_type end) : _start(0), _end(end.get()) {}

        DEVICE constexpr _Range(dim_type start, dim_type end)
            : _start(start.get()),
              _end(end.get()) {}

        DEVICE constexpr Iterator begin() const {
            return Iterator(_start);
        }

        DEVICE constexpr Iterator end() const {
            return Iterator(_end);
        }

    private:
        int _start;
        int _end;
    };

    template <typename dim_type> class _ReverseRange {
    public:
        class ReverseIterator {
        public:
            DEVICE constexpr ReverseIterator(int i) : _i(i) {}

            DEVICE dim_type operator*() const {
                return dim_type(_i);
            }

            DEVICE constexpr ReverseIterator& operator++() {
                --_i;
                return *this;
            }

            DEVICE constexpr bool operator!=(const ReverseIterator other) const {
                return _i != other._i;
            }

        private:
            int _i;
        };

        DEVICE constexpr _ReverseRange(dim_type end) : _start(end.get() - 1), _end(-1) {}

        DEVICE constexpr _ReverseRange(dim_type start, dim_type end)
            : _start(end.get() - 1),
              _end(start.get() - 1) {}

        DEVICE constexpr ReverseIterator begin() const {
            return ReverseIterator(_start);
        }

        DEVICE constexpr ReverseIterator end() const {
            return ReverseIterator(_end);
        }

    private:
        int _start;
        int _end;
    };

    template <int increment, typename dim_type>
    DEVICE constexpr auto range_with_step(dim_type end) {
        return _Range<dim_type, increment>(divup(end.get(), increment) * increment);
    }

    template <int increment, typename dim_type>
    DEVICE constexpr auto range_with_step(dim_type start, dim_type end) {
        return _Range<dim_type, increment>(
            start,
            start + static_cast<dim_type>(divup(end.get() - start.get(), increment) * increment));
    }

    template <typename dim_type> DEVICE constexpr auto range(dim_type end) {
        return _Range<dim_type>(end);
    }

    template <typename dim_type> DEVICE constexpr auto range(dim_type start, dim_type end) {
        return _Range<dim_type>(start, end);
    }

    template <typename dim_type> DEVICE constexpr auto reverse_range(dim_type end) {
        return _ReverseRange<dim_type>(end);
    }

    template <typename dim_type> DEVICE constexpr auto reverse_range(dim_type start, dim_type end) {
        return _ReverseRange<dim_type>(start, end);
    }
}

#define SPIO_DIM(Name)                                                                             \
    class Name : public spio::Dim<Name> {                                                          \
    public:                                                                                        \
        using spio::Dim<Name>::Dim;                                                                \
    }

namespace spio {
    SPIO_DIM(OFFSET);
    SPIO_DIM(LANE);
}

#endif
