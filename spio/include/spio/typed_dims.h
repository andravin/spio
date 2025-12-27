#ifndef SPIO_TYPED_DIMS_H
#define SPIO_TYPED_DIMS_H

#include "spio/meta.h"
#include "spio/dim.h"
#include "spio/compound_index.h"
#include "spio/tensor.h"
#include "spio/coordinates.h"

// Define CUDA vector types for host compilation (C++ tests)
#ifndef __CUDACC__
#ifndef SPIO_CUDA_VECTOR_TYPES_DEFINED
#define SPIO_CUDA_VECTOR_TYPES_DEFINED

struct float4 {
    float x, y, z, w;
};

struct uint4 {
    unsigned int x, y, z, w;
};

#endif // SPIO_CUDA_VECTOR_TYPES_DEFINED
#endif // __CUDACC__

#endif