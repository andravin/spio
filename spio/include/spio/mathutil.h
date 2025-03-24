#ifndef SPIO_MATHUTIL_H_
#define SPIO_MATHUTIL_H_

#include "spio/macros.h"

namespace spio
{
    template <typename T>
    DEVICE inline constexpr T min(T a, T b) { return (a < b) ? a : b; }

    template <typename T>
    DEVICE inline constexpr T max(T a, T b) { return (a > b) ? a : b; }

    DEVICE inline constexpr int divup(int a, int b) { return (a + b - 1) / b; }

    template <typename T>
    DEVICE inline void swap(T &a, T &b)
    {
        T temp = a;
        a = b;
        b = temp;
    }
}

#endif
