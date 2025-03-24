#ifndef SPIO_INDEX_BASE_H_
#define SPIO_INDEX_BASE_H_

#include "spio/macros.h"

namespace spio
{

    class IndexBase
    {
    public:
        DEVICE constexpr IndexBase(int offset = 0) : _offset(offset) {}
        DEVICE constexpr int offset() const { return _offset; }

    private:
        const int _offset;
    };
}

#endif // SPIO_INDEX_BASE_H_
