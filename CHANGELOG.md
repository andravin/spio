# **CHANGELOG.md**

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> **Note:** Some features in earlier versions were developed in private Spio branches
> prior to their first *public* release. Dates below refer to **publication dates**,
> not the dates when features were originally conceived or implemented.

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
