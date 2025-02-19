#ifndef SPIO_MACROS_H_
#define SPIO_MACROS_H_

#ifndef DEVICE
#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif
#endif

#endif
