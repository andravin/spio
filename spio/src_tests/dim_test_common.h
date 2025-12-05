#ifndef SPIO_SRC_TESTS_DIM_TEST_UTIL_H
#define SPIO_SRC_TESTS_DIM_TEST_UTIL_H

#include "utest.h"
#include "spio/typed_dims.h"

// Specialization of utest_type_deducer for Dim types
template <typename Derived> struct utest_type_deducer<spio::Dim<Derived>, false> {
    static void _(const spio::Dim<Derived>& d) {
        UTEST_PRINTF("%d", d.get());
    }
};

// Specialization of utest_type_deducer for Fold types
template <typename DimType, int Stride>
struct utest_type_deducer<spio::Fold<DimType, Stride>, false> {
    static void _(const spio::Fold<DimType, Stride>& f) {
        UTEST_PRINTF("Fold<%d>(%d)", Stride, f.get());
    }
};

template <typename DimType, int Size, int Stride>
struct utest_type_deducer<spio::Module<DimType, Size, Stride>, false> {
    static void _(const spio::Module<DimType, Size, Stride>& m) {
        UTEST_PRINTF("Module<%d, %d>(%d)", Size, Stride, m.get());
    }
};

#define UTEST_DIM_PRINTER(DimType)                                                                 \
    template <> struct utest_type_deducer<DimType, false> {                                        \
        static void _(const DimType& d) {                                                          \
            UTEST_PRINTF(#DimType "(%d)", d.get());                                                \
        }                                                                                          \
    }

#define TEST_DIM(Name)                                                                             \
    SPIO_DIM(Name);                                                                                \
    UTEST_DIM_PRINTER(Name)

UTEST_DIM_PRINTER(spio::OFFSET);

UTEST_DIM_PRINTER(spio::LANE);

#endif