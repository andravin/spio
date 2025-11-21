# Spio

Experimental CUDA kernel framework unifying typed dimensions, NVRTC JIT specialization, and ML‑guided tuning.

[![PyPI version](https://img.shields.io/pypi/v/spio.svg)](https://pypi.org/project/spio/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Overview

Spio is an experimental CUDA research playground that packages several forward-looking ideas for building next-generation GPU kernels: strongly typed tensor dimensions, pipeline-oriented code generation, and machine-learned performance models that steer NVRTC-compiled kernels at runtime.

## Key Features

### 🔧 Typed Dimension System

Unlike “Named Tensors,” which attach string names to dimensions and validate them at run time, Spio uses Typed Dimensions: each dimension is a distinct C++ type generated at build time and checked at compile time.

- Named Tensors (strings, run-time):
  - Dimension identity is a string evaluated at run time
  - Errors surface during execution
  - Requires lookups and checks in hot paths

- Typed Dimensions (types, compile-time):
  - Each logical dimension is a unique C++ type (e.g., I, J, K8)
  - Misuses fail to compile (zero run-time overhead)
  - Operator overloading maps types to per-tensor positions/strides

When the same dimension type appears in different tensors, it represents the same logical dimension; each tensor still defines its own size and stride for that dimension based on its layout. This enables position-free indexing—users don’t track index positions, sizes, or strides across tensors; the type system ensures correctness at compile time.

In practice, the generated tensor classes overload the subscript operator (e.g., operator[] and helpers like get&lt;Dim&gt;()) to accept dimension types. For each dimension type present in a tensor’s layout, the overload applies that tensor’s stride for that type; if a dimension type not used by the tensor is provided, the expression fails to compile (static_assert), with zero run-time name lookups or checks.

### ⚡ Just-in-Time Kernel Generation

Spio compiles kernels at runtime with NVIDIA’s NVRTC (libnvrtc) and tunes them for your GPU. No CUDA toolkit install is needed because Spio relies on the CUDA headers and NVRTC shared libraries that NVIDIA distributes as Python packages (the same infrastructure PyTorch depends on). And there’s no host C compiler involved at runtime—Spio invokes kernels directly through the CUDA driver API, so no generated launcher wrappers are required.

This contrasts with packages like Triton Language that require a C compiler at runtime.

### 🎯 Performance Models

Machine learning models predict optimal kernel configurations based on layer parameters and hardware characteristics. This eliminates expensive auto-tuning while achieving better performance than heuristic-based approaches.

### 🚀 PyTorch Integration

Seamless integration with PyTorch through custom operators and `torch.compile` support. Drop-in replacement for existing operations with significant speedups.

## Performance Results

### Algorithm Innovation

The cuDNN Conv2d kernels use "implicit GEMM" with 1D horizontal tiling, causing excessive memory traffic due to overlapping reads in the convolution halo. Spio uses 2D tiling with a circular-buffer overlap-add algorithm that:

- Reduces tile overlap and global memory traffic
- Maximizes register usage through loop unrolling
- Increases occupancy by minimizing local memory footprint
- Leverages Tensor Cores with 8×8 matrix operations for a group width of 8

### Benchmark Results

On NVIDIA GeForce RTX 3090, Spio approaches theoretical DRAM bandwidth limits for forward pass (FProp), input gradients (DGrad), and weight gradients (WGrad), while PyTorch/cuDNN implementations suffer from excess data transfers.

On NVIDIA GeForce RTX 4090, Spio exceeds the effective DRAM bandwidth limit for small batch sizes by effectively utilizing the 72 MB L2 cache:

![Benchmark Result on NVIDIA GeForce RTX 4090](figures/batch_size_vs_eff_bandwidth__nvidia_geforce_rtx_4090__convfirst_64c_3r_3s_8gw.png)

Benchmarks use realistic workloads with layers embedded in ConvFirst or MBConv blocks to accurately reflect real-world performance.

## Quick Start

### Prerequisites

- Linux x86_64
- NVIDIA GPU: Ampere (sm_80/sm_86) or Ada (sm_89)
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

- PyTorch (torch>=2.4.0) is an explicit dependency and will be installed automatically when you install Spio; no separate install step is required.
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

However, `torch.compile()` (Inductor/Triton) does, and missing pieces surface as errors like `nvrtc: file not found`, `error: unable to compile C wrapper`, `LLVM: external toolchain not found`, or `RuntimeError: codegen failed in Inductor`. These originate from PyTorch/Triton rather than Spio.

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

## Typed Dimensions

Spio’s typed dimensions system represents dimensions as distinct C++ types. The generator emits those types (e.g., I, J, K16, BLOCK_I), and operator overloading automatically applies dimensions subscripts with the correct stride and prevents accidental mixing of different dimensions.

A dimension type denotes a logical axis that can be used across several tensors, while each tensor provides its own size/stride for each dimension it supports. Because dimension identity is a type, operations work on specific dimension types that are known at compile time. This enables position-free indexing and aggressive compile-time optimization (constexpr indexing, loop unrolling).

The matrix multiply example first defines tensor layouts in a Python module:

```python
# Import Tensor, dtype, CompoundIndex, Fold, Dims, etc.
from spio.generators import *

# Dimension I represents the same logical dimension across all tensors that use it.
# Each tensor defines its own strided layout containing that dimension.

# Tensor A: format K16 x I x K8
# K16 and K8 are folds of dimension K with strides 16 and 8.
tensor_a = Tensor("A", dtype.uint4, Dims(k16=k16, i=m, k8=2), constant=True)

# Tensor B: format K16 x J x K8
tensor_b = Tensor("B", dtype.uint4, Dims(k16=k16, j=n, k8=2), constant=True)

# Tensor C: format J16 x I x J8
tensor_c = Tensor("C", dtype.uint4, Dims(j16=j16, i=m, j8=2))

# SmemA: format PING x K16 x I16 x CHECKERS
# I16 is a fold of dimension I with stride 16.
smem_tensor_a = Tensor(
    "SmemA",
    dtype.uint4,
    Dims(
        ping=2,
        k16=config.chunk_k16,
        i16=block_x16,
        checkers=32,
    ),
)
```

It also defines thread-block dimensions:

```python
# Dimension BLOCK_I folds dimension I with stride block_x.
Fold("block_i", "i", block_x),

# Dimension BLOCK_J folds dimension J with stride block_x.
Fold("block_j", "j", block_x),
```

Code generation defines CUDA/C++ classes for these types, and the CUDA kernel includes them from generated file "types.h".

```c++
// Include generated code.
#include "types.h"

// Dimension 'i' and folds 'block_i' and 'block_j' generate types I, BLOCK_I, and BLOCK_J
// that you use in the CUDA kernel.

// Get the tile coordinates for this thread-block.
auto block_idx = make_coordinates(BLOCK_I(blockIdx.y), BLOCK_J(blockIdx.x));

// Map this thread to the global memory load index.
GlobalLoadIndex global_load_idx(threadIdx.x);

// Map global load index into A/B tensor coordinates. Replace base dims X with I/J.
auto global_load_a_idx = global_load_idx.cast<X, I>();
auto global_load_b_idx = global_load_idx.cast<X, J>();
auto a = A(a_ptr)[block_idx][global_load_a_idx].rebase();
auto b = B(b_ptr)[block_idx][global_load_b_idx].rebase();
```

The main computation loop demonstrates how typed dimensions provide compile-time safety by preventing incompatible dimension types from being used with tensors that don't support them. The tensor implementations use `constexpr` with known tile sizes so that tensor indexing arithmetic is greatly simplified at compile-time and loops with constant bounds are unrolled. This produces highly optimized code that runs at near full utilization on NVIDIA GeForce RTX 4090 (Ada) GPUs:

```c++
// Aggressive unrolling of the main loop improves arithmetic utilization.
#pragma unroll MAIN_LOOP_UNROLL_DEPTH
for (int iter = 0; iter < size.get(); iter += 2 * step_size.get()) {

    // Double-buffer loads and compute.
    for (auto phase : range(PING(2))) {

        // If not the last iteration, copy the next tile from global
        // memory to shared memory asynchronously.
        if (iter + (phase.get() + 1) * step_size.get() < size.get()) {
            // Copy into the back-buffer.
            loader_a.copy_async(smem_a_store[(phase + 1) % 2].get(), a.get());
            loader_b.copy_async(smem_b_store[(phase + 1) % 2].get(), b.get());
        }

        // Advance the global memory tiles.
        a.step(step_size);
        b.step(step_size);

        // Synchronize on the previous iteration's global memory copy.
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();

        // Load matrix tiles from shared memory into registers.
        a_tile.load(smem_a_load[phase]);
        b_tile.load(smem_b_load[phase]);

        // Matrix-multiply the tiles using Tensor Cores.
        // Compile-time type checking ensures the compatibility of the tile dimensions.
        mma(a_tile, b_tile, c_tile, c_tile);
        __syncthreads();
    }
}
__pipeline_wait_prior(0);
```

The output staging loop demonstrates range-based iteration over all dimensions of a tensor. When the coordinate is applied to a different tensor's indexing operator, the system performs automatic **dimension normalization** (unfolding coordinates to match the target tensor's strides) and **dimension projection** (applying only matching dimension types while ignoring those not used by the target tensor):

```c++
// Iterate over c_tile; coord auto-normalizes and projects when indexing smem_c_fragment
for (auto coord : range(c_tile)) {
    *smem_c_fragment[coord] = c_tile[coord]->to_half2(f);
}
```
