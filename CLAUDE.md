# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spio (SPEE-oh) is an experimental CUDA kernel framework combining:

- **Typed dimensions** for safe, position-free tensor indexing
- **NVRTC JIT compilation** (no nvcc or host compiler required)
- **ML-guided kernel tuning** using XGBoost performance models
- **Direct CUDA Driver API execution** from Python

Supports NVIDIA Ampere (sm_80/sm_86) and Ada (sm_89) GPUs.

## Build & Test Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run tests with parallel compilation
SPIO_WORKERS=$(nproc) pytest tests

# Limit test params for faster iteration
SPIO_MAX_TEST_PARAMS=5 pytest tests/test_conv2d_gw8.py

# Run C++ tests (requires nvcc)
SPIO_ENABLE_CPP_TESTS=1 pytest -s tests/test_tutorial.py

# Filter C++ tests by glob pattern
SPIO_ENABLE_CPP_TESTS=1 SPIO_CPP_TESTS_FILTER="CursorInitializer.*" pytest -s tests/test_cpp.py::test_cpp_tests

# Quick regression check (smoke tests, excluding slow ones)
SPIO_ENABLE_CPP_TESTS=1 pytest -m "smoke and not slow"

# All smoke tests including slow ones
SPIO_ENABLE_CPP_TESTS=1 pytest -m "smoke"

# Skip slow tests entirely
pytest -m "not slow"
```

### Just Commands

The project includes a `justfile` for common tasks. Run `just` to list available commands:

| Command | Description |
|---------|-------------|
| `just test` | Run all tests with parallel compilation |
| `just test-quick` | Fast iteration (limited params) |
| `just test-fast` | Skip slow tests |
| `just smoke` | Quick regression (smoke, not slow) |
| `just smoke-all` | All smoke tests including slow |
| `just test-cpp` | Run C++ tests |
| `just test-file FILE` | Run a specific test file |

### Pytest Markers

| Marker | Purpose |
|--------|---------|
| `smoke` | Quick regression tests covering key functionality |
| `slow` | Tests that take longer to run (some overlap with smoke) |

Typical workflow: run `pytest -m "smoke and not slow"` for fast feedback during development.

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `SPIO_WORKERS` | Parallel compilation workers (default: 1) |
| `SPIO_MAX_TEST_PARAMS` | Limit kernel test configs (default: 10) |
| `SPIO_ENABLE_CPP_TESTS` | Enable nvcc-dependent tests |
| `SPIO_DEBUG` | Include CUDA debug info (-G) |
| `SPIO_LINEINFO` | Include CUDA line info |
| `SPIO_DISABLE_TF32` | Control TF32 precision |
| `SPIO_DISASM` | Dump kernel disassembly (SASS) |

## Architecture

```
spio/
├── generators/      # Python → CUDA code generation (Dim, Tensor, CompoundIndex specs)
├── kernels/         # Kernel factories, caching, performance models
├── compiler/        # NVRTC compilation pipeline
├── cuda/            # Cython bindings to CUDA Driver API (driver.pyx)
├── functional/      # PyTorch custom ops (@torch.library.custom_op)
├── layers/          # nn.Module wrappers
├── include/spio/    # C++ header-only typed dimension library
├── src/             # CUDA kernel source files (.cu)
└── src_tests/       # C++ unit tests (utest.h framework)
```

### Kernel Compilation Pipeline

1. **Generator specs** (`spio/generators/`) define dimensions, tensors, indices
2. **`generate()`** produces C++ header with type definitions
3. **NVRTC** compiles `.cu` source + generated header → cubin
4. **`KernelFactory`** caches compiled kernels per (params, config, device)
5. **CUDA Driver API** launches kernels directly from Python

## Key Patterns

### Code Generation Markers

CUDA sources use `/*@spio ... @spio*/` comment blocks for code generation:

```cpp
/*@spio
I = Dim()
K = Dim()
A = Tensor((I(16), K(32)), dtype.float)
@spio*/
```

### Kernel Factory Pattern

Each kernel type has:

- `*_params.py`: Parameters dataclass
- `*_kernel.py`: Config dataclass + `_get_kernel_spec()` function
- `KernelFactory` instance for compilation/caching

Example: `Conv2dGw8Params`, `Conv2dGw8Config`, `conv2d_gw8_kernel_factory`

### PyTorch Custom Operators

Use `@torch.library.custom_op("spio::op_name", mutates_args=())` decorator pattern.

### Memory & Precision

- Memory format: Channels-last (`torch.channels_last`) for Conv2d ops
- Compute precision: float16, accumulation: float32

## Code Style

### C++ Documentation

Minimalist, tag-free style (see `docs/spio-c++-style-guide.md`):

- Single summary sentence as first line
- Aligned parameter documentation (no Doxygen tags)
- Header guards: `SPIO_<FILENAME>_H_`

```cpp
/// Loads a tile of activations into shared memory.
///
/// Parameters:
///   gmem     Global memory pointer.
///   smem     Shared memory tile buffer.
///   idx      Tile index.
```

### Python

Standard formatting, use dataclasses for params/configs.
