# **Spio C++ Documentation Style Guide (Minimalist & Modern)**

This document defines the standard for documenting C++ and CUDA code in the Spio project (and any related work).
It emphasizes *clarity, minimalism, alignment, and readability*, without using Doxygen tags or Markdown formatting.

The goals:

* Simple, Pythonic docstrings adapted to C++
* No `\param`, `@tparam`, `@brief`, or other tag noise
* Clean vertical alignment
* Good readability in IDEs, terminals, and GitHub
* Fully compatible with modern doc extractors (optional)
* Keeps comments lightweight enough to write frequently

---

## 1. **General Principles**

### 1.1 The first line is a one-sentence summary

```cpp
/// Transposes the matrix in-place.
```

* Must be a single declarative sentence ending with a period.
* No extra formatting, no headers, no markup.

### 1.2 Leave exactly one blank comment line after the summary

```cpp
/// Transposes a matrix.
///
/// Uses an 8×8 tile and shared memory for coalesced reads.
```

### 1.3 Avoid redundancy

Do *not* repeat information that the type system already expresses.

❌ **Avoid:**
“`a: pointer to float16 data`” when the signature shows that.

✔ **Prefer:**
Only describe conceptual meaning or constraints.

### 1.4 Use simple natural language only

No Doxygen tags, no markdown bold/italic, no HTML.

---

## 2. **Aligned Argument Documentation**

Use aligned columns for function or method parameters.

### Preferred style

```cpp
/// Loads a tile of activations into shared memory.
///
/// Parameters:
///   gmem     Global memory pointer.
///   smem     Shared memory tile buffer.
///   idx      Tile index.
///
/// Returns:
///   Writes the tile into shared memory.
```

### Notes

* “Parameters:” and “Returns:” are plain text, not markup.
* The spacing between names and descriptions should be visually aligned.
* This style looks clean in monospace code and avoids ragged comment edges.

### Optional ultra-minimal style

(no “Parameters:” header)

```cpp
/// Loads a tile of activations into shared memory.
///
///   gmem     Global memory pointer.
///   smem     Shared memory tile buffer.
///   idx      Tile index.
///
/// Writes the tile into shared memory.
```

---

## 3. **Documenting Template Parameters**

Template parameters use the same aligned style.

```cpp
/// A typed dimension object used by the Typed Tensor system.
///
/// Template parameters:
///   DimType    The unique tag type for the dimension.
///   Size       Number of elements along this dimension.
///   Stride     Logical stride between successive elements.
///
/// Represents a dimension with fixed size and stride.
template<typename DimType, int Size, int Stride>
struct Dimension { ... };
```

Guidelines:

* Keep descriptions short and conceptual.
* Avoid repeating the type or default-value information from the template signature.
* Document constraints (e.g., “must be power of two”) if relevant.

---

## 4. **Documenting Classes and Structs**

Class comments describe:

* The purpose of the abstraction
* The key responsibilities
* Any invariants or constraints
* High-level conceptual information only

Example:

```cpp
/// Represents a 2D tiled view of a contiguous memory buffer.
///
/// Provides utilities for computing tile offsets, iterating over
/// blocks of memory, and performing coalesced loads into shared memory.
///
/// Template parameters:
///   T        Element type.
///   Layout   Memory layout policy.
template<typename T, typename Layout>
class TilePtr { ... };
```

Avoid describing every member variable unless they are part of the public API.

---

## 5. **Documenting CUDA Kernels**

CUDA kernel docs should focus on:

* Algorithmic behavior
* Memory assumptions
* Synchronization constraints
* Launch configuration or required block size when necessary

Example:

```cpp
/// Computes the expansion step for the fused MLP block.
///
/// Loads input tiles using cp.async, performs MMA operations using
/// Tensor Cores, and writes the hidden activations into the workspace.
///
/// Parameters:
///   in        Input activation tensor.
///   weights   Expansion weights.
///   workspace Buffer for hidden activations.
///   interval  Index of the sample interval to process.
///
/// Assumes:
///   - Input and workspace pointers are 16-byte aligned.
///   - Channel count divisible by 8.
///   - Launch uses one block per tile group.
__global__ void expand_kernel(...);
```

Keep kernel documentation strictly conceptual; do **not** describe `threadIdx.x` unless nonstandard.

---

## 6. **Documenting Inline Utilities and Free Functions**

For small helper functions, keep comments short:

```cpp
/// Returns true if the pointer is aligned to 16 bytes.
inline bool is_aligned_16(const void* p);
```

Or with a short detail paragraph:

```cpp
/// Computes the next multiple of N.
///
/// The value is rounded upward unless already a multiple of N.
inline int round_up(int value, int N);
```

---

## 7. **Documenting Operators and Overloads**

Document the *semantic meaning*, not the syntactic form.

```cpp
/// Returns a pointer to the Nth tile.
TilePtr operator+(int n) const;
```

---

## 8. **Documenting Namespaces and Files**

At file top:

```cpp
/// Utilities for memory alignment and tile size computation.
///
/// These helpers are used by both expansion and projection kernels.
```

Namespace-level docs (optional, used only in large modules):

```cpp
/// Typed tensor dimension and index utilities for Spio.
namespace spio::dims {
```

---

## 9. **Spacing and Formatting Rules**

* Align parameter names in doc comments.
* Keep comment lines ≤ ~100 characters.
* Maintain one blank line between sections.
* Avoid trailing spaces (clang-format will help).
* Use `///` for all public documentation comments.
* Use `//` for short internal comments or clarifying remarks.

---

## 10. **Examples: Complete Documentation Blocks**

## 10.1 Templated helper function

```cpp
/// Loads an M×N tile from global memory into shared memory.
///
/// Template parameters:
///   T        Element type.
///   M        Number of rows.
///   N        Number of columns.
///
/// Parameters:
///   gmem     Source pointer in global memory.
///   smem     Destination tile in shared memory.
///   idx      Tile index within the block.
///
/// Writes the tile into shared memory.
template<typename T, int M, int N>
__device__ void load_tile(const T* gmem, T* smem, int idx);
```

---

## 10.2 Typed Tensor dimension

```cpp
/// A folded dimension with a fixed stride and size.
///
/// Template parameters:
///   BaseDim     The underlying dimension type.
///   Stride      Logical stride between elements.
///   Size        Number of elements after folding.
///
/// Represents a logical dimension with a constant stride applied to
/// the underlying index space.
template<typename BaseDim, int Stride, int Size>
struct Fold { ... };
```

---

## 10.3 Class example

```cpp
/// Provides typed indexing for a tensor.
///
/// Encapsulates the mapping between linear offsets and typed
/// dimension indices. Supports unfolded and folded dimensions.
class Index {
public:
    /// Computes the linear offset for a coordinate.
    ///
    /// Parameters:
    ///   coord     A Coordinates object representing positions
    ///             along multiple typed dimensions.
    ///
    /// Returns:
    ///   The linear offset as an integer.
    int operator[](const Coordinates& coord) const;
};
```

---

## 11. **Things We Explicitly Do NOT Use**

* No `@param`, `@tparam`, `\brief`, `\details`, etc.
* No markdown bold/italic or section headers
* No HTML tags
* No table syntax
* No Doxygen block commands like `/** */` (always use `///`)
* No repetition of type information from signatures
* No trailing “Notes:” unless needed

Minimalism is the rule.

---

## 12. **Tool Compatibility Notes (Optional Section)**

You *can* use this format with:

* IDE tooltips (clangd, Visual Studio, CLion)
* Doxygen (it parses aligned text just fine)
* Sphinx + Breathe (if desired)
* mkdocs + doxygen-md
* Standardese

But the format is deliberately designed to *not depend* on any tool.
