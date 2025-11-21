#ifndef SPIO_SRC_TESTS_DIM_TEST_UTIL_H
#define SPIO_SRC_TESTS_DIM_TEST_UTIL_H

#include "utest.h"
#include "spio/dim.h"

class I : public spio::Dim<I> {
public:
    using spio::Dim<I>::Dim;
};

class J : public spio::Dim<J> {
public:
    using spio::Dim<J>::Dim;
};

class K : public spio::Dim<K> {
public:
    using spio::Dim<K>::Dim;
};

class L : public spio::Dim<L> {
public:
    using spio::Dim<L>::Dim;
};

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

template <> struct utest_type_deducer<I, false> {
    static void _(const I& d) {
        UTEST_PRINTF("I(%d)", d.get());
    }
};

template <> struct utest_type_deducer<J, false> {
    static void _(const J& d) {
        UTEST_PRINTF("J(%d)", d.get());
    }
};

template <> struct utest_type_deducer<K, false> {
    static void _(const K& d) {
        UTEST_PRINTF("K(%d)", d.get());
    }
};

template <> struct utest_type_deducer<L, false> {
    static void _(const L& d) {
        UTEST_PRINTF("L(%d)", d.get());
    }
};

#endif