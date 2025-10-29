#ifndef SPIO_STRIP_LOADER_PARAMS_H_
#define SPIO_STRIP_LOADER_PARAMS_H_

#include "spio/mathutil.h"

namespace spio
{
    template <int major_axis_size, int minor_axis_size, int num_warps>
    class StripLoaderParams
    {
    private:
        static_assert(num_warps >= major_axis_size, "num_warps must be greater than or equal to major_axis_size");
        static constexpr int _minor_axis_chunk = min(num_warps / major_axis_size, minor_axis_size);
        static_assert(minor_axis_size % _minor_axis_chunk == 0, "minor_axis_size must be a multiple of minor_axis_chunk");

    public:
        static constexpr int active_warps = _minor_axis_chunk * major_axis_size;
        static constexpr int num_loads = divup(minor_axis_size, _minor_axis_chunk);
    };
}

#endif
