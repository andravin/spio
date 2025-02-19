#ifndef SPIO_MATHUTIL_H_
#define SPIO_MATHUTIL_H_

#include "spio/macros.h"

namespace spio
{
    DEVICE inline constexpr int min(int a, int b) { return (a < b) ? a : b; }

    DEVICE inline constexpr int max(int a, int b) { return (a > b) ? a : b; }

    DEVICE inline constexpr int divup(int a, int b) { return (a + b - 1) / b; }
}

#endif
