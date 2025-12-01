#ifndef SPIO_COMPOUND_INDEX_BASE_H_
#define SPIO_COMPOUND_INDEX_BASE_H_

#include "spio/macros.h"

namespace spio {

    class CompoundIndexBase {
    public:
        DEVICE constexpr CompoundIndexBase(int offset = 0) : _offset(offset) {}

        DEVICE constexpr int offset() const {
            return _offset;
        }

    private:
        const int _offset;
    };
}

#endif
