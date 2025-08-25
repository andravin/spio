# Spio

High-performance CUDA kernels for training convolutional neural networks with PyTorch.

![Benchmark Result on NVIDIA GeForce RTX 3090](figures/batch_size_vs_eff_bandwidth__nvidia_geforce_rtx_3090__convfirst_64c_3r_3s_8gw.png)

## Overview

Spio is a framework for developing and deploying efficient GPU kernels for deep learning. While ConvNet model architectures have evolved rapidly, kernel implementations have lagged behind, often limiting training performance. Spio bridges this gap by providing tools to create kernels that approach theoretical hardware limits.

Our initial focus is grouped convolution with group width 8 and stride 1, a promising operation that has fallen into disuse due to inefficient implementations. Spio's grouped convolution kernels achieve near-optimal memory bandwidth utilization on NVIDIA Ampere and Ada GPUs.

## Key Features

### 🔧 Typed Dimension System
Spio uses a compile-time tensor specification system that generates type-safe CUDA classes for tensor indexing. This eliminates common indexing errors and makes kernel code more readable and maintainable.

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

### Integration with Timm

Use with [our timm fork](https://github.com/andravin/pytorch-image-models.git) on the `spio_dev` branch:

```bash
python train.py --model convfirst_tiny --spio
export SPIO_LOGGER=1  # Enable diagnostic output
```

## Architecture Details

### Typed Tensors

Define tensor layouts in Python:

```python
TensorSpec("Output", "uint4", {"n": n, "p": p, "q": q, "k8": c8})
IndexSpec("OutputStoreIdx", {"n": block_n, "q": block_q, "k8": block_c8})
```

Use type-safe indexing in CUDA:

```c++
OutputStoreIdx idx(threadIdx.x);
output = output.n(block_n + idx.n()).q(block_q + idx.q()).k8(block_c8 + idx.k8());
if (thread_stores_output) *output = *smem_output_load;
```

### GPU Support

- **NVIDIA Ampere**: sm_80 (A100), sm_86 (RTX 30-series)
- **NVIDIA Ada**: sm_89 (RTX 40-series)
- **Hopper support**: Coming soon

## Contributing

Spio is currently in early development. We welcome contributions from performance engineers and researchers interested in pushing the boundaries of GPU kernel efficiency.

## Citation

If you use Spio in your research, please cite our paper:

```bibtex
@article{spio2024,
  title={Efficient GPU Kernels for Convolutional Neural Networks},
  author={...},
  journal={arXiv preprint arXiv:2404.03617},
  year={2024}
}
```