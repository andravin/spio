# Spio

Experimental CUDA kernel framework unifying typed dimensions, NVRTC JIT specialization, and ML‑guided tuning.

<p align="center">
  <a href="https://pypi.org/project/spio/">
    <img src="https://img.shields.io/pypi/v/spio.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/spio/">
    <img src="https://img.shields.io/pypi/pyversions/spio.svg" alt="Python versions">
  </a>
  <img src="https://img.shields.io/pypi/wheel/spio" alt="Wheel available">
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/andravin/spio.svg" alt="License">
  </a>
</p>

## Overview

Spio is an experimental CUDA research playground that packages several forward-looking ideas for building next-generation GPU kernels: strongly typed tensor dimensions, machine-learned performance models, and direct-driver execution.

Spio compiles kernels just-in-time with NVRTC and launches them directly from Python via the CUDA Driver API. No intermediate C++ glue code, no CUDA Toolkit (`nvcc`), no host compiler (`gcc`) required.

## The Typed Dimension System

In high-performance GPU computing, memory layouts are rarely simple. We deal with swizzled shared memory, interleaved vector loads, and opaque tensor core fragments. Standard libraries try to manage this using positional indexing (e.g., `tensor(i, j, k)`), placing the cognitive load on the developer to track exactly which argument corresponds to which physical dimension.

Spio introduces a strongly typed, projective indexing system that decouples the logical description of your data from its physical layout. At its core, Spio uses a compound index to map a linear offset to logical dimensions, enabling complex geometries like tiling and swizzling to be handled transparently.

Spio implements typed dimensions in a header-only, CUDA-aware C++ library using template metaprogramming. In the following examples, the comment blocks marked with the `@spio` tag instruct Spio's code generator to pre-include header files that define the requested dimension, tensor, and compound index classes.

### 1\. Safety and Commutativity

Spio dimensions behave like integers. Because dimensions are types, it is not possible to accidentally mix different dimensions.

**File:** [01\_commutativity.cpp](spio/src_tests/tutorial/01_commutativity.cpp)

```cpp
// Define dimension types I and J.
//
/*@spio
[
Dim("i"), Dim("j")
]
@spio*/

UTEST(Lesson1, TypeSafety) {

    // Dimensions work like integers.
    EXPECT_EQ(I(2) + I(4), I(6));
    EXPECT_LT(I(8), I(10));

    // Each dimension is a different CUDA / C++ type.
    static_assert(!std::is_same_v<I, J>, "I and J are different types");

    // This would fail to compile:
    //
    // EXPECT_EQ(I(5), J(5));
    // error: no match for ‘operator==’ (operand types are ‘I’ and ‘J’)
    //
    // and so would this:
    //
    // auto sum = I(3) + J(4);
    // error: invalid operands to binary expression ('I' and 'J')
    //
    // This prevents accidental mixing of dimensions.
}
```

Spio never asks for a dimension's position in the tensor's dimensions list. Instead, Spio uses the dimension variable's static type to determine operator behavior.

For example, many frameworks implement tensor subscripting such that the position of a subscript determines its behavior. In other words, `x(i, j, k) != x(k, i, j)`. Spio enables **position-free subscripting** where `x[i][j][k] == x[k][i][j]`. The compiler determines the effect of subscripts `i`, `j`, and `k` using their static types only.

Typed dimensions also enable something we call **dimensional projection**: a coordinate list comprising many dimensions can be used as a subscript, and only dimensions supported by the tensor will have an effect, while others are ignored.

```cpp
// Define tensors A and B using dimensions I(16) × K(32) and K(32) × J(64).
//
/*@spio
[
Tensor("A", dtype.float, Dims(i=16, k=32)),
Tensor("B", dtype.float, Dims(k=32, j=64))
]
@spio*/
UTEST(Lesson1, Commutativity) {

    // Create storage for the matrices.
    A::data_type a_data[A::storage_size()];
    B::data_type b_data[B::storage_size()];

    // Create matrices a and b.
    auto a = A(a_data);
    auto b = B(b_data);

    // Verify matrix sizes.
    EXPECT_EQ(A::size<I>(), I(16));
    EXPECT_EQ(A::size<K>(), K(32));
    EXPECT_EQ(B::size<K>(), K(32));
    EXPECT_EQ(B::size<J>(), J(64));

    // Define coordinates
    auto i = I(2);
    auto j = J(3);
    auto k = K(4);

    // Position-free subscripting:
    // Subscript order does not affect the result.
    EXPECT_EQ(a[i][k].get(), a[k][i].get());
    EXPECT_EQ(b[k][j].get(), b[j][k].get());

    // Dimensional projection:
    // Coordinates project onto the tensor's supported dimensions.
    auto coords = make_coordinates(i, j, k);
    EXPECT_EQ(a[coords].get(), a[k][i].get());
    EXPECT_EQ(b[coords].get(), b[j][k].get());
}
```

### 2\. The Unbounded Cursor

Spio uses Cursors: lightweight, unbounded pointers that traverse multiple dimensions.

**File:** [02\_cursor\_movement.cpp](spio/src_tests/tutorial/02_cursor_movement.cpp)

```cpp
/*@spio
[
Tensor("A", dtype.float, Dims(i=10, j=10))
]
@spio*/

UTEST(Lesson2, AccumulationLoop) {

    // Create matrix A.
    A::data_type a_data[A::storage_size()];
    auto a = A(a_data);

    // Create cursor at (i=2, j=4).
    auto b = a[I(2)][J(4)];

    for (int step = 0; step < 5; ++step) {

        // Verify the current position.
        EXPECT_EQ(b.get(), a_data + (2 + step) * 10 + 4);

        // Step by 1 in the I dimension.
        b.step(I(1));
    }
}
```

### 3\. Folded Dimensions

The generator `Dims(k8=4, i=4, k=8)` creates a tensor with physical layout $K_8(4) \times I(4) \times K(8)$. Here, $K_8$ and $K$ together address the full logical range $K(0) \ldots K(31)$: $K_8$ selects which chunk of 8 (the quotient), and $K$ selects within that chunk (the remainder). This decomposition enables interleaved and vectorized memory layouts while letting you write loops over the logical dimension $K$.

**File:** [03\_folding.cpp](spio/src_tests/tutorial/03_folding.cpp)

```cpp
// Define a Tensor with a folded dimension K and interleaved layout.
// Layout: K8(4) x I(4) x K(8)

/*@spio
[
Tensor("A", dtype.float, Dims(k8=4, i=4, k=8))
]
@spio*/

UTEST(Lesson3, AutomaticNormalization) {

    // Create tensor a.
    A::data_type data[A::storage_size()];
    auto a = A(data);

    // Folded dimension K8 is dimension K folded by stride 8.

    // Dimensions are compatible with their folds:
    EXPECT_EQ(K8(3), K(3 * 8));
    EXPECT_EQ(K8(3) + K(4), K(3 * 8 + 4));

    // Use constant I ..
    auto i = I(2);

    // .. and loop over K in range [0 .. 31] inclusive.
    for (auto k : range(K(32))) {

        // The loop variable has type K.
        static_assert(std::is_same_v<decltype(k), K>, "k should be of type K");

        // Spio accepts logical dimension K
        // and folds it into the tensor's K8 and K dimensions automatically ..
        auto b = a[i][k];

        // .. saving the user from folding it manually.
        auto k8 = K8(k.get() / 8);
        auto km8 = K(k.get() % 8);
        auto c = a[i][k8][km8];

        EXPECT_EQ(b.get(), c.get());
    }
}
```

### 4\. Dimensional Projection

A Spio tensor acts as a filter. It accepts a world state (a superset of coordinates) and automatically projects onto the supported dimensions.

This allows you to create a single coordinates variable that includes all relevant dimensions. Each tensor projects the coordinates onto its supported dimensions, and arithmetic and comparison operators follow the same projection rules.

With dimensional projection, individual dimensions disappear from the program. Tensor definitions carry all the information about how dimensions are used, and dimensional projection automatically harvests the relevant dimensions from world coordinates.

**File:** [04\_projection.cpp](spio/src_tests/tutorial/04_projection.cpp)

```cpp
#include <numeric>

// Define tensors A, B, C, and C_tile
/*@spio
[
Tensor("A", dtype.float, Dims(i=16, k=32)),
Tensor("B", dtype.float, Dims(k=32, j=64)),
Tensor("C", dtype.float, Dims(i=16, j=64)),
Tensor("C_tile", dtype.float, Dims(i=8, j=32), Strides(i=64))
]
@spio*/
UTEST(Lesson4, DimensionalProjection) {

    // ... create tensors a, b, and c with types A, B, and C.

    // Select coordinates (I, J) for the tiles.
    //
    auto origin = spio::make_coordinates(I(12), J(60));

    // Operations on coordinates use a technique we call dimensional projection:
    // - arithmetic applies to pairs of matching dimensions and passes through others
    // - comparison tests all pairs of matching dimensions
    // - subscript applies matching dimensions and ignores others

    // For matrix a ~ I × K, subscript I matches, and J is ignored.
    auto a_tile = a[origin];

    // For matrix b ~ K × J, subscript J matches, and I is ignored.
    auto b_tile = b[origin];

    // For matrix c ~ I × J, both I and J match.
    auto c_tile = C_tile(c[origin].get());

    // Iterate over the range I(8) × J(32).
    for (auto idx : spio::range(c_tile)) {

        // Iterate over the range K(32).
        for (auto k : spio::range(a.size<K>())) {

            // local and world have dimensions (I, J, K)
            auto local = idx + k;
            auto world = origin + local;

            // Check that world coordinates I and K are less than a's sizes.
            // Ignore world coordinate J in the comparison and subscript operations.
            if (world < a.sizes()) { EXPECT_EQ(*a_tile[local], *a[world]); }

            // Check that world coordinates J and K are less than b's sizes.
            // Ignore world coordinate I in the comparison and subscript operations.
            if (world < b.sizes()) { EXPECT_EQ(*b_tile[local], *b[world]); }
        }

        // Check that world coordinates I and J are less than c's sizes.
        if (origin + idx < c.sizes()) { EXPECT_EQ(*c_tile[idx], *c[origin + idx]); }
    }
}
```

### 5. Matrix Multiply Kernel

For a full example of high-performance matrix multiply kernel using typed dimensions and just-in-time compilation, see:

- CUDA Source: [mma_checkerboard_16c.cu](spio/src_tests/mma_checkerboard_16c.cu)
- Python Generators: [test_mma_checkerboard.py](tests/matmul/test_mma_checkerboard.py)

This example demonstrates how dimensional projection manages the complexity of mapping global memory, shared memory tiles, and register matrix fragments within a single kernel.

## Additional Features

### ⚡ Just-in-Time Kernel Generation

Spio compiles kernels at runtime with NVIDIA’s NVRTC (libnvrtc) and uses a trained performance model to select the fastest kernel configuration for your GPU and workload. No CUDA toolkit install is needed because Spio relies on the CUDA headers and NVRTC shared libraries that NVIDIA distributes as Python packages (the same infrastructure PyTorch depends on). Spio launches kernels directly through the CUDA driver API, so no C/C++ launcher wrappers are required.

### 🎯 Performance Models

Machine learning models predict optimal kernel configurations based on layer parameters and hardware characteristics. This eliminates expensive auto-tuning while achieving better performance than heuristic-based approaches.

### 🚀 PyTorch Integration

Seamless integration with PyTorch through custom operators and `torch.compile` support.

## Performance Results

### Algorithm Innovation

The cuDNN Conv2d kernels use "implicit GEMM" with 1D horizontal tiling, causing excessive memory traffic due to overlapping reads in the convolution halo. Spio uses 2D tiling with a circular-buffer overlap-add algorithm that:

- Reduces tile overlap and global memory traffic
- Maximizes register usage through loop unrolling
- Increases occupancy by minimizing local memory footprint
- Leverages Tensor Cores with 8×8 matrix operations for a group width of 8

### Benchmark Results

On NVIDIA GeForce RTX 3090, Spio approaches theoretical DRAM bandwidth limits for forward pass (FProp), input gradients (DGrad), and weight gradients (WGrad), while PyTorch/cuDNN implementations suffer from excess data transfers.

On NVIDIA GeForce RTX 4090, Spio exceeds the effective DRAM bandwidth limit for small batch sizes. 2D tiling always reduces L2 traffic, and the advantage grows when inputs from the previous layer already reside in the 72 MB cache.

Benchmarks use realistic workloads with layers embedded in ConvFirst or MBConv blocks to accurately reflect real-world performance.

![Benchmark Result on NVIDIA GeForce RTX 4090](figures/batch_size_vs_eff_bandwidth__nvidia_geforce_rtx_4090__convfirst_64c_3r_3s_8gw.png)

## Quick Start

### Prerequisites

- Linux x86\_64
- NVIDIA GPU: Ampere (sm\_80/sm\_86) or Ada (sm\_89)
- NVIDIA driver (compatible with CUDA 12 runtime)
- Python 3.9+

### Installation

Create and activate a virtual environment (recommended):

```bash
python3 -m venv spio_env
source spio_env/bin/activate

# Upgrade pip.
python -m pip install --upgrade pip
```

Then install Spio from PyPI using pip:

```bash
pip install spio
```

Notes:

- PyTorch (torch\>=2.4.0) is an explicit dependency and will be installed automatically when you install Spio; no separate install step is required.
- CUDA toolkit installation is not required. Spio relies on NVIDIA's CUDA runtime and NVRTC libraries that are pulled in via wheels and are the same libraries PyTorch uses.

Alternatively, install Spio from source. For this, you will need a C compiler. On Ubuntu:

```bash
sudo apt update && sudo apt install -y build-essential
```

Then clone the Spio repository and install:

```bash
git clone https://github.com/andravin/spio.git
cd spio
pip install .

# Run tests (optional)
cd tests
SPIO_WORKERS=$(nproc) pytest .
```

Exit the virtual environment when finished.

```bash
deactivate
```

### Additional Requirements for `torch.compile()`

Spio itself does not need a host C/C++ compiler or the CUDA developer toolkit. You can use Spio operations with PyTorch on a production system that does not have these.

However, `torch.compile()` (Inductor/Triton) does, and missing pieces cause errors like `nvrtc: file not found`, `error: unable to compile C wrapper`, `LLVM: external toolchain not found`, or `RuntimeError: codegen failed in Inductor`. These originate from PyTorch/Triton rather than Spio.

If you intend to use `torch.compile()` with Spio operations, ensure your production environment provides:

- gcc or clang (or a compatible toolchain)
- CUDA driver development files (e.g., `libcuda.so` symlink or stubs)
- Optional: CUDA toolkit runtime libraries (`libnvrtc.so`, `libnvjitlink.so`, CUDA “stubs”) when GPU compilation paths require them

This recipe will add the requirements for `torch.compile()` on an Ubuntu system:

```bash
# Install development tools required by PyTorch Inductor + Triton
sudo apt update
sudo apt install -y build-essential

# Ensure the CUDA driver library has the expected unversioned symlink
# (Many cloud images only ship libcuda.so.1)
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
```

Then test:

```bash
python3 -c "import torch; torch.cuda.is_available()"
python3 -c "import torch; torch.compile(lambda x: x**2)(torch.randn(5, device='cuda'))"
```

### Usage

```python
import torch
import spio.functional

# Define input and weights for grouped convolution
x = torch.randn(32, 64, 56, 56, device='cuda', dtype=torch.float16)
weight = torch.randn(64, 8, 3, 3, device='cuda', dtype=torch.float16)

# Call the Spio custom convolution op with registered autograd support.
# Automatically selects optimal kernel configuration for your GPU. 
output = spio.functional.conv2d_gw8(x, weight, groups=8)
```
