# **CHANGELOG.md**

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> **Note:** Some features in earlier versions were developed in private Spio branches
> prior to their first *public* release. Dates below refer to **publication dates**,
> not the dates when features were originally conceived or implemented.

---

## **[0.8.0] — 2026-01-08**

### **Wave-based block traversal and loader encapsulation**

This release improves L2 cache locality through wave-based block traversal and simplifies the kernel interface through loader encapsulation. Matrix multiply performance reaches 94% tensor core utilization on RTX 4090, exceeding cuBLAS by approximately 5%.

#### Added

* Wave-based block traversal for improved L2 cache hit rate (60% → 82%).
* Inline PTX `cp.async.cg.shared.global.L2::128B` replaces `__pipeline_memcpy_async` intrinsic.
* `TwoFold` tensor format for matrix layouts with two fold dimensions.
* Dynamic shared memory with configurable carveout (100% default).
* Extended `MmaConfig`: `warps_m`, `warps_n`, `wave_size`, `chunk_k16=4`.
* Support for 128×256 and 256×128 block tiles.
* Four-warp matrix multiply configurations with 2D async strip loader.
* Per-load bounds masking for async strip loaders.
* `prefetch_async()` method for initial tile loads before main loop.
* Benchmark CLI: `--warp-m`, `--warp-n`, `--warps-m`, `--warps-n`, `--chunk-k16`, `--unroll-depth`, `--wave-size`, `--pytorch`.
* Predicates in disasm output for SASS analysis.
* `Tensor.__getitem__` as synonym for `initializer()`.
* Support for CUDA 13.x (runtime detection of CUDA version from PyTorch).

#### Changed

* Loaders now own both smem and global cursors internally.
* `copy_async(phase)` replaces `copy_async(smem_phase, global_phase)`.
* `CursorInitializer` now creates `Cursor` subclass.
* Global matrix layout changed to I16 × K16 × I × K8.
* Swap inner/outer loop nesting in `AsyncStripLoader2D` for better address locality.
* Rename `derive_dim` to `with_dim`.
* Split derived dimension protocols: `DerivedDimension` and `SizedDerivedDimension`.
* Eliminate one `__syncthreads()` from main loop.
* Remove explicit nvidia-cuda-* dependencies; PyTorch manages CUDA package versions.

#### Fixed

* Add `__restrict__` to global memory pointers in loaders.

### Performance Models

* No changes to convolution kernels.
* Performance model archives from 0.7.x are compatible.

---

## **[0.7.1] — 2025-12-28**

### **Python 3.9 compatibility fix**

#### Fixed

* Type annotation `Path | None` incompatible with Python 3.9.

#### Tests

* Fix `test_warp_semaphore()` dtype for older CUDA array interface compatibility.

---

## **[0.7.0] — 2025-12-28**

### **Derived dimensions, tensor initializers, and orthogonal dimension addition**

This release introduces derived dimensions, allowing tensors to encode nonlinear coordinate
transformations (like swizzle patterns) directly into their definitions. Swizzle patterns
and load indices that previously required manual setup in kernel code are now folded into
tensor definitions by the generators.

#### Added

* Derived dimensions: tensors can now include dimensions that transform coordinates
  (e.g., checkerboard swizzle patterns) as part of their type definition.
* Tensor initializers (`Tensor.initializer()`): factory functions that automatically apply
  index subscripts when constructing tensor views.
* Tensor copy construction: factory methods that construct tensors from compatible base tensors
  in the same derivation chain.
* `Tensor.with_vector_length()`: create tensor views with modified vector length.
* `Cursor.inbounds()`: bounds checking method for cursor positions.
* `Cursor.extents()`: access tensor extents from cursor.
* Orthogonal dimension addition: `I(1) + J(2)` now produces `make_coordinates(I(1), J(2))`,
  enabling `a[i][j][k] == a[i + j + k] == a[j][k][i]`.
* `Coordinates` generator for explicit coordinate list generation.
* `CompoundIndexPartition` generator for cooperative iteration patterns.
* Anonymous generator naming: generators without explicit names receive automatic names.
* `Generators` class: container that automatically sets generator names from attribute names.
* Assignment syntax for `@spio` blocks: `A = Tensor(...)` instead of `Tensor("A", ...)`.
* Builtin CUDA expressions (`threadIdx.x`, etc.) as fold and compound index initializers.
* `dtype.half8` as synonym for `dtype.uint4`.
* `get_dtype_veclen()` and `get_dtype_with_veclen()` utilities.
* `Tensor.num_bytes()` method.
* `SPIO_COUNT_INSTRUCTIONS` environment variable and `count_instructions` context variable.
* `SPIO_DISASM` environment variable for printing disassembly.
* `disasm()` function using nvdisasm.

#### Changed

* Renamed `ComputeIndex` to `LocalIndex`.
* DimInfo now uses plain `int` for size and stride instead of `_OffsetDim` type.
* Added `BaseDim` base class unifying `Dim`, `Fold`, and `Module`.
* `Dim` now has a trivial `unfold()` method returning self.
* Simplified cross-type arithmetic using common implementations with operation lambdas.
* Simplified double-buffer indexing in matmul example to use implicit modulo wrap via
  dimensional projection.
* Allocate shared memory as `char` array instead of `uint4`.
* Moved compiler environment variables to `spio.compiler.flags`.
* Extracted CUDA path detection to `spio.compiler.cuda_paths`.
* Updated README examples to use assignment syntax in `@spio` blocks.
* Clarified torch.compile requirements section with specific error message.

#### Removed

* `get_base_dim_type_t`, `get_dim_stride`, `fold_to_target()` helpers (superseded by
  uniform `unfold()` and `fold()` methods).
* `_OffsetDim` type (replaced by plain `int`).
* `test_row_memcpy_kernel()` (functionality covered elsewhere).

### Performance Models

* No changes to convolution kernels.
* Performance model archives from 0.6.0 are fully compatible.

---

## **[0.6.0] — 2025-12-09**

### **Compound index projection, deferred folding, and documentation improvements**

This release enhances the compound index system with projection capabilities,
fixes a fundamental issue with cross-fold carry in Cursor, adds automatic fold
size inference, and substantially improves documentation.

#### Added

* Builtin `LANE` and `OFFSET` dimensions for compound index projection.
* `partition()` static method on `CompoundIndex` for cooperative iteration.
* Constructors for `CompoundIndex` from `Coordinates` and compatible dimensions.
* Automatic fold size inference: use `-1` in `Dims()` to derive size from fold ratios.
* Tensor-like access for fragment classes: `as_tensor()`, subscript operators, `range()` iteration.
* `Tensor::extent()` and `Tensor::extents()` methods for hierarchical bounds checking.
* Compound index tutorial (lesson 5).
* Stride validation: dimension names in `Strides()` must match tensor dimensions.

#### Changed

* `Cursor` now uses `Coordinates` internally instead of an integer offset.
  This enables correct cross-fold carry: `a[e][f][g] == a[e + f + g]`.
* Renamed `Tensor::sizes()` to `Tensor::extents()`.
* Renamed fragment `CompoundIndex` type to `compound_index_type`.
* Standardized naming conventions in the matrix multiply example.
* Replaced "checkers" terminology with "swizzle" in matmul example.

#### Fixed

* Wrong stride dimension name in `conv2d_gw8_wgrad` kernel caused shared memory
  bank conflicts in some configurations.

#### Documentation

* Rewrote introduction to The Typed Dimension System section.
* Added efficiency note: abstractions resolve at compile time with no runtime overhead.
* Added Limitations section: tensor dimensions are compile-time constants.
* Improved Performance Models section with concrete XGBoost details.
* Added compound index tutorial to README.

### Performance Models

* No changes to convolution kernels.
* Performance model archives from 0.5.0 are fully compatible.

---

## **[0.5.0] – 2025-12-04**

### **Major expansion of the Typed Dimensions & projection system**

This release substantially enhances the Typed Dimensions system and introduces
a new tutorial, expanded dimensional-projection capabilities, and improvements
across coordinates, folds, modules, indexing, and generator infrastructure.

#### Added

* `SPIO_DIM` macro for defining dimension types.
* New header that consolidates all typed-dimension features.
* Comprehensive Typed Dimensions tutorial with multiple examples.
* Dimension printers for C++ tests.
* Extraction of generator specs from `@spio`-tagged comments.
* `tutorial.h` header for C++ tutorials.
* `pre_includes` and `run_args` options to `compile_with_nvcc`.
* `CPP_TESTS_FILTER` environment variable for filtering C++ unit tests.
* Binary operations between dim-like types and `Coordinates`.
* `is_dim_like`, `dims_compatible_v`, `is_bounded_v`, and related metaprogramming traits.
* `get_dim_stride`, `get_dim_size`, `tuple_element`, and improved meta-utilities.
* Fast path for `Cursor::operator[]` when stride matches target dimension.

#### Changed

* Normalized dimension and coordinate handling throughout the projection system.
* Consolidated generator includes into `spio/typed_dims.h`.
* Converted all dimension names to uppercase at initialization.
* Renamed `index.py` → `compound_index.py`.
* Improved coordinate folding, matching, projection, and stride logic.
* Modified `CompoundIndex::get()` to return a size-aware `Module`.
* Updated `_Range` iterator return types for stronger typing.
* Improved `Cursor::operator[]` to accept both indices and objects with `to_coordinates()`.

#### Removed

* `apply_to(Tensor)` entry points (superseded by improved coordinate logic).
* Obsolete `_ConstantRowColIndex`.

#### Documentation

* Updated README with revised Typed Dimensions section.
* Added tutorial examples for dimensional projection and folding.

### Performance Models

* No changes to convolution kernels.

* Performance model archives from 0.4.2 are fully compatible.

---

## **[0.4.2] – 2025-11-18**

### **Improved runtime CUDA loading, documentation, and generator usability**

#### Added

* Runtime loading of `libcuda.so.1` with function pointer dispatch.
* Error checking for maximum number of CUDA kernel launch arguments.
* Notes and requirements for `torch.compile()` usage.
* `__all__` support in `spio.generators` for cleaner imports.
* Additional documentation for conv2d and generator usage.

#### Changed

* Updated installation instructions.
* Updated version to **0.4.2**.
* Switched `import spio.generators as gen` to `from spio.generators import *`.
* Ensured `LaunchParams` always uses 3-tuple grid/block types.
* Improved formatting and docstrings throughout the package.

---

## **[0.4.1] – 2025-11-16**

### **Perf-model compatibility and packaging cleanup**

#### Added

* Logic ensuring performance-model version matching ignores patch versions.
* PyPI installation instructions.

#### Changed

* Updated project metadata.
* Reformatted `pyproject.toml`.
* Changed version to **0.4.1**.

---

## **[0.4.0] – 2025-11-07**

### **First public release including the Typed Dimensions system**

This version is the first **published** release of Spio to include the Typed
Dimensions system and the generalized tensor-multiply infrastructure.
These capabilities were developed earlier as part of the Spio project and were
published here for the first time.

#### Added

* Typed Dimensions framework (`Dim`, `Fold`, `Module`, etc.).
* Generalized tensor-multiply infrastructure.
* Typed-dimension support for grouped convolution kernels.
* New CUDA grouped conv kernels using typed dimensions.
* Fallback logic to use `libcuda.so.1` automatically.
* New device architecture entries (e.g., A100 SXM4).
* Updated and expanded README documentation.

#### Changed

* Bumped version to **0.4.0**.
* `timm` dependency is now optional and imported only when used.

---

## **[0.3.x] – 2024**

### Early public releases

These versions contained the foundational Spio features prior to the
introduction and later publication of the Typed Dimensions system.

#### Highlights

* Added PyPI metadata.
* Initial `parse_dataclass()` implementation (later rewritten).
* Early grouped-convolution kernels and JIT compilation framework.
* Base Python + Cython + CUDA infrastructure.
