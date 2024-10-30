# Spio

Efficient CUDA kernels for training convolutional neural networks with PyTorch.

![Benchmark Result on NVIDIA GeForce RTX 3090](figures/batch_size_vs_eff_bandwidth__nvidia_geforce_rtx_3090__convfirst_64c_3r_3s_8gw.png)

## Introduction

The goal of the Spio project is to improve training efficiency for convolutional neural networks (ConvNets). While there has been a lot of progress in the design of ConvNet models, the performance of ConvNet kernels has languished. Today, the performance of a ConvNet is often limited by the efficiency of its implementation.

Our [paper](https://arxiv.org/abs/2404.03617) implemented efficient GPU kernels for ConvNet inference. Spio implements kernels for training.

The first Spio kernel is for grouped convolution, a promising layer that has fallen into disuse because of the inefficiency of the current implementation. We focus on group width equal to eight and stride 1, as used in our ConvFirst model, and support NVIDIA Ampere ([sm_80](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) and [sm_86](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)) and Ada ([sm_89](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf)) GPUs.

### Audience

At this early stage of development, Spio is for performance engineers and other heroes. As we add more kernels, Spio will guide model researchers to safety, like the Nereid Spio guiding sailors through treacherous waters.

## Benchmarks

The cuDNN Conv2d kernels use an "implicit GEMM" algorithm that tiles the input tensor with horizontal strips. The support halo for the convolution kernel causes overlapping reads of the input tensor, and when the tile is a 1D strip, the overlap is larger than the tile. This results in excess global memory traffic.

The Spio Conv2d kernel uses 2D tiles. This reduces the overlap between tiles and reduces global memory traffic. It processes the 2D tile one row at a time, convolving each input row with every filter row while updating a circular buffer of output rows. The circular buffer is implemented in registers by unrolling the input-row loop by the number of filter rows. This overlap-add style algorithm minimizes the kernel's local memory footprint, which increases occupancy and maximizes utilization of the global memory bandwidth.

Group width 8 matches the accumulation depth of the Float16 tensor core (through AD102, sm_89). Therefore, the grouped convolution is implemented just like regular planar convolution, but with scalar input elements
replaced by 8-element vectors, scalar filter elements replaced by 8x8 matrices, and scalar multiplication replaced by matrix-vector multiplication. Processing 16 columns of the input row at once turns the input vectors into input matrices, so that the algorithm can use the [mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma) instruction.

On the NVIDIA RTX 3090 GPU (above), Spio approaches the DRAM memory bandwidth limit for the FProp, DGrad (gradient with respect to inputs), and WGrad (gradient with respect to weights) kernels, while the PyTorch / cuDNN kernels struggle with excess data transfers.

On the NVIDIA RTX 4090 GPU, Spio exceeds the DRAM memory bandwidth limit for small batch sizes by exploiting the fact that the activation tensors fit in the GPU's large (72 MB) L2 cache:

![Benchmark Result on NVIDIA GeForce RTX 4090](figures/batch_size_vs_eff_bandwidth__nvidia_geforce_rtx_4090__convfirst_64c_3r_3s_8gw.png)

### Benchmarking Methodology

Our benchmarks use [torch.profile](https://pytorch.org/docs/stable/profiler.html), which uses NVIDIA's [libcupti](https://developer.nvidia.com/cupti-ctk12_0) internally for precise
kernel timing. We benchmark layers *in situ*, placing a grouped convolution layer inside a
ConvFirst or MBConv building block and constructing a stack of several blocks. This creates a realistic environment for the target kernel, where the memory hierarchy is exercised similarly to a real-world use case.

## Implementation Notes

Spio uses several strategies to simplify the development of high-performance CUDA kernels that
integrate with PyTorch.

### Named Tensors

Spio uses named tensors to simplify tensor indexing in CUDA source code. In Python, you specify the tensor
and indexing dimensions like this:

```python
        TensorSpec("Output", "uint4", {"n": n, "p": p, "q": q, "k8": c8}),
        TensorSpec(
            "ConstSmemOutput",
            "const uint4",
            {"q": block_q, "n": block_n, "k8": block_c8 + 1},
        ),
        IndexSpec("OutputStoreIdx", {"n": block_n, "q": block_q, "k8": block_c8}),
```

which generates CUDA/C++ classes that you use in your kernel like this:

```c++
    // Output-smem to output.
    ConstSmemOutput smem_output_load(smem_output_buf);
    Output output(dst);
    bool thread_stores_output;
    {
        OutputStoreIdx idx(threadIdx.x);
        auto q = block_q + idx.q();
        auto n = block_n + idx.n();
        auto k8 = block_c8 + idx.k8();
        smem_output_load = smem_output_load.n(idx.n()).q(idx.q()).k8(idx.k8());
        output = output.n(n).p(block_p).q(q).k8(k8);
        thread_stores_output = n < Output::N && q < Output::Q && k8 < Output::K8 &&
            threadIdx.x < OutputStoreIdx::size;
    }

    # ...

    if (thread_stores_output)
    {
        *output = *smem_output_load;
    }
    output = output.p(1);

```

### Run Time Compilation

Spio compiles kernels at runtime using [libnvrtc](https://docs.nvidia.com/cuda/nvrtc/index.html) and launches them with [libcuda](https://docs.nvidia.com/cuda/cuda-driver-api/index.html). Unlike other packages that offer runtime compilation, Spio does not depend on the CUDA toolkit. We simply use the same NVIDIA [libnvrtc](https://pypi.org/project/nvidia-cuda-nvrtc-cu12/) and [cuda-runtime](https://pypi.org/project/nvidia-cuda-runtime-cu12/) Python packages on which PyTorch already [depends](https://github.com/pytorch/pytorch/blob/bae3426af77be643af83f1527fb430e9ca09b058/.github/scripts/generate_binary_build_matrix.py#L71). This minimizes software dependencies and simplifies installation.

### Kernel Performance Models

Spio predicts the best kernel configuration for each layer with a performance model trained on thousands of offline benchmarking samples. Prediction takes just a few milliseconds, so startup is much faster than other frameworks that use a time consuming auto-tuning step.

### Integration with torch.compile

We integrate with `torch.compile` using the [Python Custom Operators](https://pytorch.org/tutorials/advanced/python_custom_ops.html) interface from PyTorch 2.4. This functionality passes basic tests but is still experimental. See this [PyTorch issue](https://github.com/pytorch/pytorch/issues/137033).

## Installation from Source

First, ensure you have a C compiler installed. On Ubuntu:

```bash
sudo apt update
sudo apt install build-essential
```

Clone the repository:

```bash
git clone https://github.com/andravin/spio.git
cd spio
```

Optionally, create a virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package from source using pip:

```bash
pip install --upgrade pip
pip install .
```

Optionally, run the unit tests. This can take a while,
because Spio tests every configuration of each kernel. It goes a bit faster
if we set the SPIO_WORKERS environment variable to use all CPU cores for compiling kernels:

```bash
cd tests
SPIO_WORKERS=$(nproc) pytest .
```

Note: the tests and scripts cannot be run from the top-level spio directory because
that would cause Python to find the local spio package instead of the installed package.
Only the installed package includes the compiled spio.cuda.driver Cython extension, so using
the local package would result in an import error. Therefore, running `cd tests` before `pytest .` is essential.

## Using Spio with Timm

Spio is integrated with [our fork](https://github.com/andravin/pytorch-image-models.git) of pytorch-image-models (timm) on the `spio_dev` branch. Add the `--spio` option to the command line of `benchmark.py`, `validate.py`, or `train.py`, and timm will use the Spio implementation for any supported operations.

Set the environment variable `export SPIO_LOGGER=1` to cause Spio to print diagnostic info to the console.
