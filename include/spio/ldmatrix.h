#ifndef SPIO_LDMATRIX_H_
#define SPIO_LDMATRIX_H_

namespace spio
{
   __device__ unsigned ldmatrix_x1(const void * p) {
        size_t s = __cvta_generic_to_shared(p);
        unsigned x;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16"
            " {%0}, [%1];"
            : "=r"(x)
            : "l"(s)
        );
        return x;
    }

   __device__ uint2 ldmatrix_x2(const void * p) {
        size_t s = __cvta_generic_to_shared(p);
        uint2 v;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16"
            " {%0, %1}, [%2];"
            : "=r"(v.x), "=r"(v.y)
            : "l"(s)
        );
        return v;
    }

   __device__ uint4 ldmatrix_x4(const void * p) {
        size_t s = __cvta_generic_to_shared(p);
        uint4 v;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            " {%0, %1, %2, %3}, [%4];"
            : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
            : "l"(s)
        );
        return v;
    }
}

#endif
