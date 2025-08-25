# Spio

High-performance CUDA kernels for training convolutional neural networks with PyTorch.

![Benchmark Result on NVIDIA GeForce RTX 3090](figures/batch_size_vs_eff_bandwidth__nvidia_geforce_rtx_3090__convfirst_64c_3r_3s_8gw.png)

## Overview

Spio is a framework for developing and deploying efficient GPU kernels for deep learning. While ConvNet model architectures have evolved rapidly, kernel implementations have lagged behind, often limiting training performance. Spio bridges this gap by providing tools to create kernels that approach theoretical hardware limits.

Our initial focus is grouped convolution with group width 8 and stride 1, a promising operation that has fallen into disuse due to inefficient implementations. Spio's grouped convolution kernels achieve near-optimal memory bandwidth utilization on NVIDIA Ampere and Ada GPUs.

## Key Features

### 🔧 Typed Dimension System
Spio uses a compile-time tensor specification system that generates type-safe CUDA classes for tensor indexing. This eliminates common indexing errors and makes kernel code more readable and maintainable.

Each dimension type identifies a unique logical dimension that can be used across multiple tensors. When the same dimension type appears in different tensors, it represents the same logical dimension, while each tensor defines its own size and stride for that dimension based on its layout.

### ⚡ Just-in-Time Kernel Generation
Kernels are compiled at runtime using NVIDIA's libnvrtc, automatically optimized for your specific GPU architecture. No CUDA toolkit installation required - Spio uses the same NVIDIA libraries that PyTorch already depends on.

### 🎯 Performance Models
Machine learning models predict optimal kernel configurations based on layer parameters and hardware characteristics. This eliminates expensive auto-tuning while achieving better performance than heuristic-based approaches.

### 🚀 PyTorch Integration
Seamless integration with PyTorch through custom operators and torch.compile support. Drop-in replacement for existing operations with significant speedups.

## Performance Results

### Algorithm Innovation

The cuDNN Conv2d kernels use "implicit GEMM" with 1D horizontal tiling, causing excessive memory traffic due to overlapping reads in the convolution halo. Spio uses 2D tiling with a circular buffer overlap-add algorithm that:

- Reduces tile overlap and global memory traffic
- Maximizes register usage through loop unrolling
- Increases occupancy by minimizing local memory footprint
- Leverages tensor cores with 8x8 matrix operations for group width 8

### Benchmark Results

On NVIDIA RTX 3090, Spio approaches theoretical DRAM bandwidth limits for forward pass (FProp), input gradients (DGrad), and weight gradients (WGrad), while PyTorch/cuDNN implementations suffer from excess data transfers.

On NVIDIA RTX 4090, Spio exceeds DRAM bandwidth limits for small batch sizes by effectively utilizing the 72MB L2 cache:

![Benchmark Result on NVIDIA GeForce RTX 4090](figures/batch_size_vs_eff_bandwidth__nvidia_geforce_rtx_4090__convfirst_64c_3r_3s_8gw.png)

Benchmarks use realistic workloads with layers embedded in ConvFirst or MBConv blocks to accurately reflect real-world performance.

## Quick Start

### Installation

```bash
# Install system dependencies (Ubuntu)
sudo apt update && sudo apt install build-essential

# Clone and install
git clone https://github.com/andravin/spio.git
cd spio
pip install .

# Run tests (optional)
cd tests
SPIO_WORKERS=$(nproc) pytest .
```

### Usage

```python
import torch
import spio

# Replace PyTorch grouped convolution
x = torch.randn(32, 64, 56, 56, device='cuda', dtype=torch.float16)
weight = torch.randn(64, 8, 3, 3, device='cuda', dtype=torch.float16)

# Automatic kernel selection and compilation
output = spio.grouped_conv2d(x, weight, groups=8)
```

## Architecture Details

### Typed Tensors

Spio's typed tensor system provides compile-time safety and readability for multi-dimensional indexing through **operator overloading on dimension types**. Each dimension type identifies a unique logical dimension that can be used across multiple tensors. When the same dimension type appears in different tensors, it represents the same logical dimension, while each tensor defines its own size and stride for that dimension based on its layout. This enables **index-position free** indexing where users don't need to track dimension index positions, sizes, or strides across different tensors.

Define tensor layouts in Python:

```python
# The 'i' dimension type represents the same logical dimension across all tensors
# But each tensor defines its own size and stride for 'i' based on its layout
tensor_a = gen.Tensor(
    "A", gen.dtype.uint4, 
    gen.Dims(k16=k16, i=m, k8=2),  # 'i' is at position 1 with size m
    constant=True
)
smem_tensor_a = gen.Tensor(
    "SmemA", gen.dtype.uint4,
    gen.Dims(ping=2, k16=config.chunk_k16, i16=block_x16, checkers=32)  # 'i16' at position 2 with size block_x16
)
tensor_c = gen.Tensor(
    "C", gen.dtype.uint4, 
    gen.Dims(i=m, j8=n8)  # 'i' is at position 0 with size m
)
```

In traditional CUDA code, you manually track array indices and remember that `A[k][i][k8]` corresponds to `C[i][j8]`. With Spio's operator overloading, the same dimension type automatically maps to the correct position and stride in each tensor:

```c++
// Index-position free indexing using operator overloading
// The dimension type 'i' automatically maps to the correct position and stride in each tensor

auto global_i = block_i.unfold() + global_load_idx.get<X>().cast<I>();

// Same 'i' dimension type works correctly across different tensors
// - In tensor A: 'i' maps to position 1 with A's stride for dimension 1
// - In tensor C: 'i' maps to position 0 with C's stride for dimension 0
auto a_element = A(a_ptr)[global_i][global_load_idx.get<K8>()];  
auto c_element = C(c_ptr)[global_i];                            

// The user doesn't track positions, sizes, or strides - the type system handles it all
// Type safety prevents dimension misuse at compile time
// auto wrong = smem_a[compute_idx.get<WARP_J>()];  // Compile error: WARP_J not valid for SmemA
```

Real example from [mma_checkerboard_16c.cu](spio/src_tests/mma_checkerboard_16c.cu):

```c++
// Multi-dimensional indexing with automatic position mapping and stride calculation
auto smem_c_store = smem_c[compute_idx.get<WARP_I>()]     // WARP_I maps to its position in smem_c
                          [compute_idx.get<WARP_J>().fold<8>()]  // WARP_J.fold<8> maps to j8 dimension
                          [c_idx.get<J2M4>().cast<J2>()]         // Type conversion and mapping
                              .rebase();

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
- **Index-position free operations**: No need to track array positions, sizes, or strides manually
- **Type safety**: Prevents using wrong dimension types at compile time

### GPU Support

- **NVIDIA Ampere**: sm_80 (A100), sm_86 (RTX 30-series)
- **NVIDIA Ada**: sm_89 (RTX 40-series)
