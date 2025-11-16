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

In practice, the generated tensor classes overload the indexing operator (e.g., operator[] and helpers like get&lt;Dim&gt;()) to accept dimension types. For each dimension type present in a tensor’s layout, the overload applies that tensor’s stride for that type; if a dimension type not used by the tensor is provided, the expression fails to compile (static_assert), with zero run-time name lookups or checks.

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

Spio’s typed dimensions system represents dimensions as distinct C++ types (not run-time strings). The generator emits those types (e.g., I, J, K16, BLOCK_I), and kernels use operator overloading to map them to the correct position and stride per tensor. The same dimension type denotes the same logical axis across tensors, while each tensor provides its own size/stride. Because dimension identity is a type, mistakes are caught at compile time, with no run-time name lookups or checks. This is what enables index-position-free indexing and aggressive compile-time optimization (constexpr indexing, loop unrolling).

Operator overloading details:

- The generated tensor classes define typed indexing (operator[] chains and get&lt;Dim&gt;() helpers) that accept dimension types in any order and compute offsets using that tensor’s per-dimension strides.
- If you pass a dimension type that the tensor does not declare, the code fails to compile via static_assert, preventing invalid indexing from reaching run time.

Define tensor layouts for a matrix multiply kernel in the Python generator:

```python
# Import Tensor, dtype, Index, Fold, Dims, etc.
from spio.generators import *

# Dimension 'i' represents the same logical dimension across all tensors
# But each tensor defines its own size and stride for 'i' based on its layout
tensor_a = Tensor(
    "A", dtype.uint4, 
    # Dimension 'i' is at position 1 with size m
    Dims(k16=k16, i=m, k8=2),
    constant=True
)
smem_tensor_a = Tensor(
    "SmemA", dtype.uint4,
    # Fold dimension 'i' with stride 16 at position 2 with size block_x16
    Dims(ping=2, k16=config.chunk_k16, i16=block_x16, checkers=32)  
)
tensor_c = Tensor(
    "C", dtype.uint4,
    # Dimension 'i' is at position 0 with size m 
    Dims(i=m, j8=n8)
)
global_load_index = Index("GlobalLoadIndex", Dims(x16=block_x16, x=16, k8=2))


# Define additional tensors for the CUDA kernel...
```

Define thread-block tiles in Python:

```python
# Dimension 'block_i' folds dimension 'i' with stride block_x.
Fold("block_i", "i", block_x)

# Dimension 'block_j' folds dimension 'j' with stride block_x.
Fold("block_j", "j", block_x)
```

In traditional CUDA code, you manually track array indices and remember that `A[k][i][k8]` corresponds to `C[i][j8]`. With Spio's operator overloading, the same dimension type automatically maps to the correct position and stride in each tensor:

```c++
// Include generated code.
#include "parameters.h"

// Dimension 'i' and folds 'block_i' and 'block_j' generate types I, BLOCK_I, and BLOCK_J
// that you use in the CUDA kernel.

// Map thread-block coordinates to blocks of I and J.
BLOCK_I block_i(blockIdx.y);

// Map the thread index to our tensor's global coordinates X16, X, and K8.
GlobalLoadIndex global_load_idx(threadIdx.x);

// Add the block and thread coordinates to compute this thread's I-coordinate.
auto global_i = block_i.unfold() + global_load_idx.get<X>().cast<I>();

// Same 'i' dimension type works correctly across different tensors
// - In tensor A: 'i' maps to position 1 with A's stride for dimension 1
// - In tensor C: 'i' maps to position 0 with C's stride for dimension 0
auto a_element = A(a_ptr)[global_i][global_load_idx.get<K8>()];  
auto c_element = C(c_ptr)[global_i];                            

// The user doesn't track positions, sizes, or strides - the type system handles it all
// Type safety prevents dimension misuse at compile time (e.g., using WARP_J with SmemA would fail to compile)
```

The main computation loop demonstrates how typed dimensions provide compile-time safety by preventing incompatible dimension types from being used with tensors that don't support them. The tensor implementations use `constexpr` with known tile sizes so that tensor indexing arithmetic is greatly simplified at compile-time and loops with constant bounds are unrolled. This produces highly optimized code that runs at near full utilization on NVIDIA GeForce RTX 4090 (Ada) GPUs:

```c++
// Main computation loop with pipelined memory operations
for (int iter = 0; iter < size.get(); iter += 2 * step_size.get())
{
    // Double-buffer loads and compute.
    for (auto phase : range(PING(2)))
    {
        // If not the last iteration, load the next tile from global
        // memory to shared memory asynchronously.
        if (iter + (phase.get() + 1) * step_size.get() < size.get())
        {
            // Load into the back-buffer.
            loader_a.load(smem_a_store[(phase + 1) % 2].get(), a.get());
            loader_b.load(smem_b_store[(phase + 1) % 2].get(), b.get());
        }

        // Advance the global memory tiles.
        a.step(step_size); 
        b.step(step_size);

        // Synchronize on the previous iteration's global memory load.
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();

        // Load matrix tiles from shared memory.
        a_tile.load(smem_a_load[phase]);
        b_tile.load(smem_b_load[phase]);

        // Matrix-multiply the tiles using Tensor Cores.
        // Compile-time type checking ensures the compatibility of the tile dimensions.
        mma(a_tile, b_tile, c_tile, c_tile);
        __syncthreads();
    }
}
```

The output staging loop demonstrates how dimensions can be dynamically refolded with different strides, while the type system ensures compile-time safety by preventing incompatible fold operations:

```c++
// Nested loops using typed dimension iterators - no manual index calculations
for (auto i16 : range(c_tile.size<I16>())) {
    for (auto j16 : range(c_tile.size<J16>())) {
        *smem_c_cursor[j16.fold<8>()][i16] = c_tile[i16][j16]->to_half2(f);
    }
}
```

The system automatically handles:

- **Logical dimension consistency**: Same dimension type represents the same logical dimension across all tensors
- **Automatic position mapping**: Operator overloading maps dimension types to correct array positions
- **Per-tensor size and stride**: Each tensor defines its own size and stride for shared dimensions
- **Index-position-free operations**: No need to track array positions, sizes, or strides manually
- **Type safety**: Prevents using wrong dimension types at compile time
- **Memory layout optimization**: Automatic padding and alignment
