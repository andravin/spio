"""Unit tests for Tensor construction patterns."""

from spio.generators import (
    Generators,
    Tensor,
    CompoundIndex,
    Dim,
    FragmentBase,
    FragmentType,
    dtype,
    Dims,
    Strides,
    Checkerboard,
    generate,
)


class TestTensorBasicConstruction:
    """Test basic Tensor construction patterns."""

    def test_dtype_and_keyword_dims(self):
        """Tensor with dtype and keyword-style Dims."""
        tensor = Tensor(dtype.float, Dims(i=16, j=32))

        g = Generators()
        g.MyTensor = tensor

        code = generate(g)
        assert "using MyTensor" in code
        assert "struct I" in code
        assert "struct J" in code

    def test_dtype_and_dict_dims(self):
        """Tensor with dtype and dict for dims (legacy style)."""
        tensor = Tensor(dtype.float, dims={"i": 16, "j": 32})

        g = Generators()
        g.MyTensor = tensor

        code = generate(g)
        assert "using MyTensor" in code

    def test_positional_dtype_and_dims(self):
        """Tensor with positional dtype and Dims arguments."""
        tensor = Tensor(dtype.half, Dims(warp=4, lane=32, i=8))

        g = Generators()
        g.T = tensor

        code = generate(g)
        assert "using T" in code
        assert "struct WARP" in code
        # LANE is a built-in dim, so it's referenced but not generated as a struct
        assert "LANE" in code

    def test_vector_dtype(self):
        """Tensor with vector dtype like half8."""
        tensor = Tensor(dtype.half8, Dims(i=16, j=4))

        g = Generators()
        g.VecTensor = tensor

        code = generate(g)
        assert "using VecTensor" in code
        # half8 is mapped to uint4 in CUDA
        assert "uint4" in code

    def test_constant_tensor(self):
        """Tensor marked as constant."""
        tensor = Tensor(dtype.float, Dims(i=16), constant=True)

        g = Generators()
        g.ConstTensor = tensor

        code = generate(g)
        assert "const float" in code


class TestTensorWithStaticDims:
    """Test Tensor construction with StaticDim objects."""

    def test_static_dim_objects(self):
        """Tensor with Dim objects called to create StaticDim."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.A = Tensor(dtype.float, Dims(g.I(16), g.K(32)))

        code = generate(g)
        assert "using A" in code
        assert "struct I" in code
        assert "struct K" in code

    def test_unnamed_dim_objects(self):
        """Tensor with unnamed Dim objects (lazy name resolution)."""
        I = Dim()
        K = Dim()
        tensor = Tensor(dtype.float, Dims(I(16), K(32)))

        g = Generators()
        g.I = I
        g.K = K
        g.MyTensor = tensor

        code = generate(g)
        assert "struct I" in code
        assert "struct K" in code
        assert "using MyTensor" in code

    def test_anonymous_dims(self):
        """Tensor with anonymous Dims (auto-generated names)."""
        I = Dim()
        J = Dim()
        tensor = Tensor(dtype.float, Dims(I(16), J(32)))

        g = Generators()
        g.MyTensor = tensor  # Only tensor is named

        code = generate(g)
        assert "struct _ANONDIM" in code
        assert "using MyTensor" in code


class TestTensorWithFolds:
    """Test Tensor construction with Fold dimensions."""

    def test_static_fold(self):
        """Tensor with StaticFold-based Dims."""
        g = Generators()
        g.K = Dim()
        g.K8 = g.K.fold(8)
        g.Matrix = Tensor(dtype.float, Dims(g.K(8), g.K8(4)))

        code = generate(g)
        assert "struct K" in code
        assert "spio::Fold<K, 8>" in code
        assert "using Matrix" in code

    def test_fold_with_auto_size(self):
        """Tensor with fold and auto-computed size (-1)."""
        K = Dim()
        K8 = K.fold(8)
        # K8 has stride 8, size 4 -> K should auto-compute to fill the gap
        # The auto-size computation gives K the remaining elements within K8's range
        tensor = Tensor(dtype.float, Dims(K8(4), K(-1)))

        g = Generators()
        g.K = K
        g.K8 = K8
        g.MyTensor = tensor

        # K is the inner dimension within each K8 group (stride 8 / stride 1 = 8 elements)
        dims_dict = dict(tensor.dims.items())
        assert dims_dict["K8"] == 4
        assert dims_dict["K"] == 8  # K fills within each K8 group

    def test_anonymous_fold(self):
        """Tensor with anonymous fold (auto-generated name)."""
        K = Dim("K")  # Named dim
        K16 = K.fold(16)  # Anonymous fold
        tensor = Tensor(dtype.float, Dims(K16(4), K(-1)))

        g = Generators()
        g.MyTensor = tensor

        code = generate(g)
        assert "struct K" in code
        assert "using _ANONFOLD" in code
        assert "spio::Fold<K, 16>" in code


class TestTensorWithDerivedDims:
    """Test Tensor construction with derived dimensions."""

    def test_division_derived_dim(self):
        """Tensor with dimension derived by division."""
        g = Generators()
        g.I = Dim()
        # i / 16 creates a SizedDerivedDimension
        g.A = Tensor(dtype.float, Dims(g.I(64) / 16, g.I(64) % 16))

        code = generate(g)
        assert "using A" in code

    def test_checkerboard_derived_dim(self):
        """Tensor with checkerboard swizzle pattern."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.K8 = g.K.fold(8)
        g.ASmem = Tensor(
            dtype.half8,
            Dims(
                g.K8(4),
                g.I(16),
                swizzle=Checkerboard(pairs_dim=g.I, colors_dim=g.K8, size=32),
            ),
        )

        code = generate(g)
        assert "using ASmem" in code


class TestTensorWithStrides:
    """Test Tensor construction with explicit strides."""

    def test_explicit_stride(self):
        """Tensor with explicitly specified stride."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.J8 = g.J.fold(8)
        g.Matrix = Tensor(
            dtype.half2,
            Dims(g.I(32), g.J8(16)),
            strides=Strides(g.J8(68)),  # Custom stride for padding
        )

        code = generate(g)
        assert "using Matrix" in code
        # The stride should appear in the generated code
        assert "68" in code

    def test_keyword_strides(self):
        """Tensor with keyword-style strides."""
        tensor = Tensor(
            dtype.float,
            Dims(i=16, j=32),
            strides=Strides(i=64),  # Explicit stride for i
        )

        g = Generators()
        g.MyTensor = tensor

        code = generate(g)
        assert "using MyTensor" in code


class TestTensorWithFragmentType:
    """Test Tensor construction with FragmentBase data types."""

    def test_fragment_as_dtype(self):
        """Tensor with FragmentBase as data type."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.AFragment = FragmentBase(FragmentType.M16_K16_F16_A, g.I, g.K)
        # FragmentBase tensor dimensions are typically the number of fragments
        g.AReg = Tensor(g.AFragment, Dims(g.K(32) / 16, g.I(64) / 16))

        code = generate(g)
        assert "using AReg" in code
        assert "AFragment" in code


class TestTensorDerivedMethods:
    """Test methods that create new Tensors from existing ones."""

    def test_vector_length(self):
        """Tensor.vector_length() creates tensor with wider vector type."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.Base = Tensor(dtype.half2, Dims(g.I(16), g.J(8)))
        g.Wide = g.Base.vector_length(8)

        code = generate(g)
        assert "using Base" in code
        assert "using Wide" in code
        # half8 is mapped to uint4 in CUDA
        assert "uint4" in code

    def test_with_dim(self):
        """Tensor.with_dim() adds derived dimension."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.K8 = g.K.fold(8)
        g.AFragment = FragmentBase(FragmentType.M16_K16_F16_A, g.I, g.K)
        g.ASmem = Tensor(dtype.half8, Dims(g.K8(4), g.I(16)))
        g.ALoadSmem = g.ASmem.with_dim(g.AFragment.load_index)

        code = generate(g)
        assert "using ASmem" in code
        assert "using ALoadSmem" in code


class TestCursorInitializer:
    """Test Tensor.initializer() and subscript syntax."""

    def test_initializer_single_index(self):
        """Tensor.initializer() with single CompoundIndex."""
        g = Generators()
        g.AGlobal = Tensor(dtype.half, Dims(warp=4, lane=32, i=8))
        g.ALoadIndex = CompoundIndex(Dims(warp=4, lane=32))
        g.AGlobalLoader = g.AGlobal.initializer(g.ALoadIndex)

        code = generate(g)
        assert "struct AGlobalLoader : AGlobal::cursor_type" in code
        assert "ALoadIndex()" in code

    def test_initializer_multiple_indices(self):
        """Tensor.initializer() with multiple indices."""
        g = Generators()
        g.AGlobal = Tensor(dtype.half, Dims(warp=4, lane=32, i=8))
        g.WarpIdx = CompoundIndex(Dims(warp=4))
        g.LaneIdx = CompoundIndex(Dims(lane=32))
        g.AGlobalLoader = g.AGlobal.initializer(g.WarpIdx, g.LaneIdx)

        code = generate(g)
        assert "[WarpIdx()][LaneIdx()]" in code

    def test_subscript_syntax_single(self):
        """Tensor[index] subscript syntax."""
        g = Generators()
        g.AGlobal = Tensor(dtype.half, Dims(i=16, j=32))
        g.Idx = CompoundIndex(Dims(i=16))
        g.AGlobalLoader = g.AGlobal[g.Idx]

        code = generate(g)
        assert "struct AGlobalLoader" in code

    def test_subscript_syntax_multiple(self):
        """Tensor[idx1, idx2] subscript syntax."""
        g = Generators()
        g.AGlobal = Tensor(dtype.half, Dims(i=16, j=32, k=8))
        g.Idx1 = CompoundIndex(Dims(i=16))
        g.Idx2 = CompoundIndex(Dims(j=32))
        g.AGlobalLoader = g.AGlobal[g.Idx1, g.Idx2]

        code = generate(g)
        assert "[Idx1()][Idx2()]" in code


class TestTensorProperties:
    """Test Tensor property accessors."""

    def test_size(self):
        """Tensor.size returns number of elements for storage."""
        tensor = Tensor(dtype.float, Dims(i=16, j=32, k=8))

        g = Generators()
        g.T = tensor

        # Force initialization
        _ = generate(g)

        # size is based on outer dimension size * stride
        assert tensor.size == 16 * 32 * 8

    def test_num_bytes(self):
        """Tensor.num_bytes returns total byte size."""
        tensor = Tensor(dtype.float, Dims(i=16, j=32))  # float = 4 bytes

        g = Generators()
        g.T = tensor

        # Force initialization
        _ = generate(g)

        assert tensor.num_bytes == 16 * 32 * 4

    def test_num_bytes_vector_type(self):
        """Tensor.num_bytes with vector dtype."""
        tensor = Tensor(dtype.half8, Dims(i=16, j=4))  # half8 = 16 bytes

        g = Generators()
        g.T = tensor

        # Force initialization
        _ = generate(g)

        assert tensor.num_bytes == 16 * 4 * 16


class TestTensorEdgeCases:
    """Test edge cases and error handling."""

    def test_1d_tensor(self):
        """Single dimension tensor."""
        tensor = Tensor(dtype.float, Dims(i=128))

        g = Generators()
        g.Vec = tensor

        code = generate(g)
        assert "using Vec" in code

    def test_many_dimensions(self):
        """Tensor with many dimensions."""
        tensor = Tensor(dtype.float, Dims(n=4, h=32, w=64, c=128))

        g = Generators()
        g.BigTensor = tensor

        code = generate(g)
        assert "using BigTensor" in code

    def test_shared_dims_across_tensors(self):
        """Multiple tensors sharing the same Dim objects."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.K = Dim()
        g.A = Tensor(dtype.float, Dims(g.I(16), g.K(32)))
        g.B = Tensor(dtype.float, Dims(g.K(32), g.J(64)))
        g.C = Tensor(dtype.float, Dims(g.I(16), g.J(64)))

        code = generate(g)

        # Each dim should appear only once
        assert code.count("struct I") == 1
        assert code.count("struct J") == 1
        assert code.count("struct K") == 1
        # All tensors should be generated
        assert "using A" in code
        assert "using B" in code
        assert "using C" in code


class TestTensorPositionalDimArgs:
    """Use the positional args for dims."""

    def test_many_dimensions_positional(self):
        """Tensor with many dimensions using positional args."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.T = Tensor(g.I(16), g.J(32), data_type=dtype.float)

        code = generate(g)
        assert "using T" in code

    def test_single_dim_positional(self):
        """Tensor with single dimension as positional arg."""
        g = Generators()
        g.I = Dim()
        g.Vec = Tensor(g.I(128), data_type=dtype.float)

        code = generate(g)
        assert "using Vec" in code
        assert "struct I" in code

    def test_three_dims_positional(self):
        """Tensor with three dimensions as positional args."""
        g = Generators()
        g.N = Dim()
        g.H = Dim()
        g.W = Dim()
        g.Image = Tensor(g.N(4), g.H(32), g.W(64), data_type=dtype.half)

        code = generate(g)
        assert "using Image" in code
        assert "struct N" in code
        assert "struct H" in code
        assert "struct W" in code

    def test_positional_with_fold(self):
        """Tensor with StaticFold as positional arg."""
        g = Generators()
        g.K = Dim()
        g.K8 = g.K.fold(8)
        g.T = Tensor(g.K8(4), g.K(-1), data_type=dtype.float)

        code = generate(g)
        assert "using T" in code
        assert "spio::Fold<K, 8>" in code

    def test_positional_with_derived_dim(self):
        """Tensor with derived dimension (division) as positional arg."""
        g = Generators()
        g.I = Dim()
        g.T = Tensor(g.I(64) / 16, g.I(64) % 16, data_type=dtype.float)

        code = generate(g)
        assert "using T" in code

    def test_positional_with_constant(self):
        """Tensor with positional dims and constant=True."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.ConstT = Tensor(g.I(16), g.J(32), data_type=dtype.float, constant=True)

        code = generate(g)
        assert "const float" in code

    def test_positional_with_strides(self):
        """Tensor with positional dims and explicit strides."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.T = Tensor(
            g.I(16),
            g.J(32),
            data_type=dtype.float,
            strides=Strides(g.I(64)),
        )

        code = generate(g)
        assert "using T" in code
        assert "64" in code  # Custom stride

    def test_positional_with_vector_dtype(self):
        """Tensor with positional dims and vector dtype."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.T = Tensor(g.I(16), g.J(4), data_type=dtype.half8)

        code = generate(g)
        assert "using T" in code
        assert "uint4" in code  # half8 maps to uint4

    def test_positional_unnamed_dims(self):
        """Tensor with unnamed Dim objects as positional args."""
        I = Dim()
        J = Dim()
        tensor = Tensor(I(16), J(32), data_type=dtype.float)

        g = Generators()
        g.I = I
        g.J = J
        g.T = tensor

        code = generate(g)
        assert "using T" in code
        assert "struct I" in code
        assert "struct J" in code

    def test_positional_anonymous_dims(self):
        """Tensor with anonymous (never named) dims as positional args."""
        I = Dim()
        J = Dim()
        tensor = Tensor(I(16), J(32), data_type=dtype.float)

        g = Generators()
        g.T = tensor  # Only tensor is named, dims get auto-generated names

        code = generate(g)
        assert "using T" in code
        assert "struct _ANONDIM" in code

    def test_positional_mixed_dims_and_folds(self):
        """Tensor with mix of StaticDim and StaticFold as positional args."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.K8 = g.K.fold(8)
        g.T = Tensor(g.I(16), g.K8(4), g.K(-1), data_type=dtype.half)

        code = generate(g)
        assert "using T" in code
        assert "struct I" in code
        assert "struct K" in code
        assert "spio::Fold<K, 8>" in code

    def test_positional_with_fragment_dtype(self):
        """Tensor with positional dims and FragmentBase as data type."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.AFragment = FragmentBase(FragmentType.M16_K16_F16_A, g.I, g.K)
        g.AReg = Tensor(g.K(32) / 16, g.I(64) / 16, data_type=g.AFragment)

        code = generate(g)
        assert "using AReg" in code
        assert "AFragment" in code

    def test_positional_shared_dims_multiple_tensors(self):
        """Multiple tensors sharing Dim objects using positional syntax."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.K = Dim()
        g.A = Tensor(g.I(16), g.K(32), data_type=dtype.float)
        g.B = Tensor(g.K(32), g.J(64), data_type=dtype.float)
        g.C = Tensor(g.I(16), g.J(64), data_type=dtype.float)

        code = generate(g)

        # Each dim should appear only once
        assert code.count("struct I") == 1
        assert code.count("struct J") == 1
        assert code.count("struct K") == 1
        assert "using A" in code
        assert "using B" in code
        assert "using C" in code


class TestTensorTupleDimArgs:
    """Test Tensor construction with tuple for dimension arguments."""

    def test_tuple_basic_dims(self):
        """Tensor with tuple of StaticDims."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.T = Tensor((g.I(16), g.J(32)), data_type=dtype.float)

        code = generate(g)
        assert "using T" in code
        assert "struct I" in code
        assert "struct J" in code

    def test_tuple_with_derived_dims(self):
        """Tensor with tuple containing derived dimensions (division/modulo)."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        i = g.I(64)
        k = g.K(32)
        g.T = Tensor((i / 16, k / 16, i % 16, k % 16), data_type=dtype.half)

        code = generate(g)
        assert "using T" in code

    def test_tuple_with_folds(self):
        """Tensor with tuple containing StaticFold dimensions."""
        g = Generators()
        g.K = Dim()
        g.K8 = g.K.fold(8)
        g.T = Tensor((g.K8(4), g.K(-1)), data_type=dtype.float)

        code = generate(g)
        assert "using T" in code
        assert "spio::Fold<K, 8>" in code

    def test_tuple_mixed_dims_and_derived(self):
        """Tensor with tuple mixing StaticDim and derived dimensions."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        i = g.I(128)
        j = g.J(64)
        g.T = Tensor((i / 16, j / 16, i % 16, (j % 16) / 8), data_type=dtype.half8)

        code = generate(g)
        assert "using T" in code

    def test_tuple_with_constant(self):
        """Tensor with tuple dims and constant=True."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.T = Tensor((g.I(16), g.K(32)), data_type=dtype.half, constant=True)

        code = generate(g)
        assert "const __half" in code

    def test_tuple_with_strides(self):
        """Tensor with tuple dims and explicit strides."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.T = Tensor(
            (g.I(16), g.J(32)),
            data_type=dtype.float,
            strides=Strides(g.I(64)),
        )

        code = generate(g)
        assert "using T" in code
        assert "64" in code


class TestTensorDictAliasedDims:
    """Test Tensor construction with dict for aliased dimensions (e.g., swizzle)."""

    def test_dict_swizzle_checkerboard(self):
        """Tensor with dict containing swizzle=Checkerboard."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.K8 = g.K.fold(8)
        g.ASmem = Tensor(
            (g.K8(4), g.I(16), dict(swizzle=Checkerboard(g.I, g.K8, size=32))),
            data_type=dtype.half8,
        )

        code = generate(g)
        assert "using ASmem" in code
        # Checkerboard generates a CheckerboardIndex type
        assert "Checkerboard" in code or "SWIZZLE" in code

    def test_tuple_with_inline_checkerboard(self):
        """Tensor with tuple containing Checkerboard directly via dict."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.J8 = g.J.fold(8)
        i_block = g.I(128)
        k16_chunk = 2
        g.T = Tensor(
            (
                g.J8(k16_chunk * 2),
                i_block / 16,
                dict(swizzle=Checkerboard(g.I, g.J8, size=32)),
            ),
            data_type=dtype.half8,
        )

        code = generate(g)
        assert "using T" in code

    def test_multiple_tensors_with_checkerboard(self):
        """Multiple tensors each with their own checkerboard swizzle."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.K = Dim()
        g.K8 = g.K.fold(8)

        i_block = g.I(128)
        j_block = g.J(128)
        k16_chunk = g.K8(2)

        g.ASmem = Tensor(
            (
                k16_chunk * 2,
                i_block / 16,
                dict(swizzle=Checkerboard(g.I, g.K8, size=32)),
            ),
            data_type=dtype.half8,
        )
        g.BSmem = Tensor(
            (
                k16_chunk * 2,
                j_block / 16,
                dict(swizzle=Checkerboard(g.J, g.K8, size=32)),
            ),
            data_type=dtype.half8,
        )

        code = generate(g)
        assert "using ASmem" in code
        assert "using BSmem" in code

    def test_dict_at_different_positions(self):
        """Dict can appear at any position in the tuple."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.K8 = g.K.fold(8)

        # Dict in middle position
        g.T1 = Tensor(
            (g.K8(4), dict(swizzle=Checkerboard(g.I, g.K8, size=32)), g.I(16)),
            data_type=dtype.half8,
        )

        code = generate(g)
        assert "using T1" in code


class TestTensorMmaCheckerboardPatterns:
    """Test patterns from the mma_checkerboard kernel for realistic usage."""

    def test_global_memory_tensor_pattern(self):
        """Global memory tensor pattern from mma_checkerboard."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()

        m = 8192
        k = 1024
        i = g.I(m)
        k_dim = g.K(k)

        # Pattern: (i / 16, k / 16, i % 16, (k % 16) / 8)
        g.AGlobal = Tensor(
            (i / 16, k_dim / 16, i % 16, (k_dim % 16) / 8),
            data_type=dtype.half8,
            constant=True,
        )

        code = generate(g)
        assert "using AGlobal" in code
        assert "const" in code

    def test_shared_memory_tensor_with_checkerboard(self):
        """Shared memory tensor with checkerboard swizzle from mma_checkerboard."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.K8 = g.K.fold(8)

        i_block = g.I(128)
        k16_chunk = 2

        g.ASmem = Tensor(
            (
                g.K8(k16_chunk * 2),
                i_block / 16,
                dict(swizzle=Checkerboard(g.I, g.K8, size=32)),
            ),
            data_type=dtype.half8,
        )

        code = generate(g)
        assert "using ASmem" in code

    def test_output_tensor_with_strides(self):
        """Output tensor with custom strides for bank conflict avoidance."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.J8 = g.J.fold(8)

        m_warps = 4
        n_warps = 2
        m_warp = 64
        n_warp = 32
        m_block = m_warp * m_warps
        n_block = n_warp * n_warps

        # i_warps and j_warps are derived dimensions (divisions)
        i_warps = g.I(m_block) / m_warp
        j_block = g.J(n_block)
        i_warp = g.I(m_warp)

        g.CSmem = Tensor(
            (i_warps, j_block / 8, i_warp, (j_block % 8) / 2),
            data_type=dtype.half2,
            strides=Strides(g.J8((m_warp + 1) * 4)),
        )

        code = generate(g)
        assert "using CSmem" in code

    def test_register_tensor_with_fragment_dtype(self):
        """Register tensor with FragmentBase as data type."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.AFragment = FragmentBase(FragmentType.M16_K16_F16_A, g.I, g.K)

        k16_chunk = 2
        m_warp = 64

        g.AReg = Tensor(
            (g.K(k16_chunk * 16) / 16, g.I(m_warp) / 16),
            data_type=g.AFragment,
        )

        code = generate(g)
        assert "using AReg" in code
        assert "AFragment" in code

    def test_full_matmul_tensor_set(self):
        """Full set of tensors like in mma_checkerboard kernel."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.K = Dim()
        g.K8 = g.K.fold(8)

        # Problem sizes
        m, n, k = 8192, 1024, 1024
        m_block, n_block = 128, 128
        k16_chunk = 2

        i = g.I(m)
        j = g.J(n)
        k_dim = g.K(k)
        i_block = g.I(m_block)
        j_block = g.J(n_block)

        # Global tensors
        g.AGlobal = Tensor(
            (i / 16, k_dim / 16, i % 16, (k_dim % 16) / 8),
            data_type=dtype.half8,
            constant=True,
        )
        g.BGlobal = Tensor(
            (j / 16, k_dim / 16, j % 16, (k_dim % 16) / 8),
            data_type=dtype.half8,
            constant=True,
        )

        # Shared memory tensors with checkerboard
        g.ASmem = Tensor(
            (
                g.K8(k16_chunk * 2),
                i_block / 16,
                dict(swizzle=Checkerboard(g.I, g.K8, size=32)),
            ),
            data_type=dtype.half8,
        )
        g.BSmem = Tensor(
            (
                g.K8(k16_chunk * 2),
                j_block / 16,
                dict(swizzle=Checkerboard(g.J, g.K8, size=32)),
            ),
            data_type=dtype.half8,
        )

        code = generate(g)
        assert "using AGlobal" in code
        assert "using BGlobal" in code
        assert "using ASmem" in code
        assert "using BSmem" in code


class TestTensorCheckerboardWithOffsetDim:
    """Test Checkerboard with offset_dim parameter for direct positional usage."""

    def test_checkerboard_with_offset_dim(self):
        """Checkerboard with offset_dim specified directly (no dict wrapper)."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.K8 = g.K.fold(8)
        g.SWIZZLE = Dim()

        i_block = g.I(128)
        k16_chunk = 2

        # Using offset_dim instead of dict(swizzle=...)
        g.ASmem = Tensor(
            (
                g.K8(k16_chunk * 2),
                i_block / 16,
                Checkerboard(g.I, g.K8, size=32, offset_dim=g.SWIZZLE),
            ),
            data_type=dtype.half8,
        )

        code = generate(g)
        assert "using ASmem" in code
        assert "SWIZZLE" in code

    def test_checkerboard_offset_dim_shared_across_tensors(self):
        """Multiple tensors sharing the same offset_dim for Checkerboard."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.K = Dim()
        g.K8 = g.K.fold(8)
        g.SWIZZLE = Dim()

        i_block = g.I(128)
        j_block = g.J(128)
        k16_chunk = 2

        # Both tensors use the same SWIZZLE dimension
        g.ASmem = Tensor(
            (
                g.K8(k16_chunk * 2),
                i_block / 16,
                Checkerboard(g.I, g.K8, size=32, offset_dim=g.SWIZZLE),
            ),
            data_type=dtype.half8,
        )
        g.BSmem = Tensor(
            (
                g.K8(k16_chunk * 2),
                j_block / 16,
                Checkerboard(g.J, g.K8, size=32, offset_dim=g.SWIZZLE),
            ),
            data_type=dtype.half8,
        )

        code = generate(g)
        assert "using ASmem" in code
        assert "using BSmem" in code
        assert "SWIZZLE" in code

    def test_mixed_offset_dim_and_dict_style(self):
        """Mix of offset_dim style and dict style in same generator set."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.K = Dim()
        g.K8 = g.K.fold(8)
        g.SWIZZLE = Dim()

        i_block = g.I(128)
        j_block = g.J(128)
        k16_chunk = 2

        # ASmem uses offset_dim style
        g.ASmem = Tensor(
            (
                g.K8(k16_chunk * 2),
                i_block / 16,
                Checkerboard(g.I, g.K8, size=32, offset_dim=g.SWIZZLE),
            ),
            data_type=dtype.half8,
        )

        # BSmem uses dict style (swizzle alias)
        g.BSmem = Tensor(
            (
                g.K8(k16_chunk * 2),
                j_block / 16,
                dict(swizzle=Checkerboard(g.J, g.K8, size=32)),
            ),
            data_type=dtype.half8,
        )

        code = generate(g)
        assert "using ASmem" in code
        assert "using BSmem" in code
        assert "SWIZZLE" in code

    def test_checkerboard_offset_dim_string(self):
        """Checkerboard with string offset_dim (e.g., 'LANE')."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.K8 = g.K.fold(8)

        i_block = g.I(128)
        k16_chunk = 2

        # Using string offset_dim (built-in dimension)
        g.ASmem = Tensor(
            (
                g.K8(k16_chunk * 2),
                i_block / 16,
                Checkerboard(g.I, g.K8, size=32, offset_dim="LANE"),
            ),
            data_type=dtype.half8,
        )

        code = generate(g)
        assert "using ASmem" in code
        assert "LANE" in code

    def test_full_mma_pattern_with_offset_dim(self):
        """Full mma_checkerboard pattern using offset_dim style."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.K = Dim()
        g.SWIZZLE = Dim()
        g.K8 = g.K.fold(8)

        # Problem sizes
        m, n, k = 8192, 1024, 1024
        m_block, n_block = 128, 128
        k16_chunk = 2

        i = g.I(m)
        j = g.J(n)
        k_dim = g.K(k)
        i_block = g.I(m_block)
        j_block = g.J(n_block)

        # Global tensors
        g.AGlobal = Tensor(
            (i / 16, k_dim / 16, i % 16, (k_dim % 16) / 8),
            data_type=dtype.half8,
            constant=True,
        )
        g.BGlobal = Tensor(
            (j / 16, k_dim / 16, j % 16, (k_dim % 16) / 8),
            data_type=dtype.half8,
            constant=True,
        )

        # Shared memory tensors with checkerboard using offset_dim
        g.ASmem = Tensor(
            (
                g.K8(k16_chunk * 2),
                i_block / 16,
                Checkerboard(g.I, g.K8, size=32, offset_dim=g.SWIZZLE),
            ),
            data_type=dtype.half8,
        )
        g.BSmem = Tensor(
            (
                g.K8(k16_chunk * 2),
                j_block / 16,
                dict(swizzle=Checkerboard(g.J, g.K8, size=32)),
            ),
            data_type=dtype.half8,
        )

        code = generate(g)
        assert "using AGlobal" in code
        assert "using BGlobal" in code
        assert "using ASmem" in code
        assert "using BSmem" in code
        assert "SWIZZLE" in code
