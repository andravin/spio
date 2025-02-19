#ifndef SPIO_PIPELINE_H_
#define SPIO_PIPELINE_H_

#include "spio/macros.h"

namespace spio
{
    class Pipeline
    {
    public:
        DEVICE Pipeline(unsigned state = 0) : _state(state) {}

        DEVICE bool active(unsigned stage) const { return (stage & _state) != 0; }

        DEVICE void step(bool active)
        {
            _state <<= 1;
            _state |= (active ? 1 : 0);
        }

    private:
        unsigned _state;
    };
} // namespace spio

#endif // SPIO_PIPELINE_H_
