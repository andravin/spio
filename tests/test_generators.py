"""Unit tests for the Generators class."""

import pytest

from spio.generators import (
    Generators,
    Tensor,
    CompoundIndex,
    Dim,
    Fold,
    Fragment,
    FragmentType,
    Macro,
    Matmul,
    dtype,
    Dims,
    Checkerboard,
    generate,
)

pytestmark = pytest.mark.smoke


class TestGeneratorsBasic:
    """Test basic Generators functionality."""

    def test_empty_generators(self):
        """An empty Generators container should have length 0."""
        g = Generators()
        assert len(g) == 0
        assert list(g) == []

    def test_assign_tensor(self):
        """Assigning a Tensor should set its class_name."""
        g = Generators()
        g.MyTensor = Tensor(dtype.float, Dims(i=16, j=32))

        assert g.MyTensor.class_name == "MyTensor"
        assert len(g) == 1
        assert "MyTensor" in g

    def test_assign_compound_index(self):
        """Assigning a CompoundIndex should set its class_name."""
        g = Generators()
        g.BlockIndex = CompoundIndex(Dims(i16=32, j16=32))

        assert g.BlockIndex.class_name == "BlockIndex"
        assert len(g) == 1

    def test_assign_dim(self):
        """Assigning a Dim should set its dim_name."""
        g = Generators()
        g.X = Dim()

        assert g.X.dim_name == "X"
        assert g.X.class_name == "X"

    def test_assign_fold(self):
        """Assigning a Fold should set its fold_name."""
        g = Generators()
        g.block_i = Fold("i", 64)

        assert g.block_i.fold_name == "BLOCK_I"

    def test_assign_fragment(self):
        """Assigning a Fragment should set its class_name."""
        g = Generators()
        g.AFragment = Fragment(FragmentType.M16_K16_F16_A, "i", "k")

        assert g.AFragment.class_name == "AFragment"

    def test_assign_macro(self):
        """Assigning a Macro should work (no-op for class_name)."""
        g = Generators()
        g.macros = Macro(dict(UNROLL_DEPTH=""))

        assert len(g) == 1
        assert "macros" in g

    def test_assign_matmul(self):
        """Assigning a Matmul should set its function_name."""
        g = Generators()
        g.AReg = Tensor(dtype.float, Dims(i=4, k=4))
        g.BReg = Tensor(dtype.float, Dims(k=4, j=4))
        g.CReg = Tensor(dtype.float, Dims(i=4, j=4))
        g.mma = Matmul(g.AReg, g.BReg, g.CReg, g.CReg)

        assert g.mma.function_name == "mma"


class TestGeneratorsIteration:
    """Test iteration over Generators."""

    def test_iterate_values(self):
        """Iterating should yield generators (like a list)."""
        g = Generators()
        g.A = Tensor(dtype.float, Dims(i=16))
        g.B = Tensor(dtype.float, Dims(j=32))

        items = list(g)
        assert len(items) == 2
        assert items[0].class_name == "A"
        assert items[1].class_name == "B"

    def test_keys(self):
        """keys() should return generator names."""
        g = Generators()
        g.A = Tensor(dtype.float, Dims(i=16))
        g.B = Tensor(dtype.float, Dims(j=32))

        keys = list(g.keys())
        assert keys == ["A", "B"]

    def test_values(self):
        """values() should return generators."""
        g = Generators()
        g.A = Tensor(dtype.float, Dims(i=16))

        values = list(g.values())
        assert len(values) == 1
        assert values[0].class_name == "A"

    def test_list_conversion(self):
        """list() should convert Generators to a list of generators."""
        g = Generators()
        g.A = Tensor(dtype.float, Dims(i=16))
        g.B = CompoundIndex(Dims(i=4, j=4))

        specs = list(g)
        assert len(specs) == 2
        assert specs[0].class_name == "A"
        assert specs[1].class_name == "B"


class TestGeneratorsNonGeneratorValues:
    """Test handling of non-generator values."""

    def test_private_attribute(self):
        """Private attributes should be stored normally."""
        g = Generators()
        g._my_private = 42

        assert g._my_private == 42
        assert len(g) == 0  # Not in registry

    def test_non_generator_stored_as_attribute(self):
        """Non-generator values should be stored as regular attributes."""
        g = Generators()
        g.my_config = {"param": 123}

        assert g.my_config == {"param": 123}
        assert len(g) == 0  # Not in registry


class TestGeneratorsCodeGeneration:
    """Test that Generators integrates with code generation."""

    def test_generate_with_generators(self):
        """generate() should accept a Generators object."""
        g = Generators()
        g.A = Tensor(dtype.float, Dims(i=16, j=32))
        g.B = CompoundIndex(Dims(i=4, j=4))

        code = generate(g)

        assert "using A" in code
        assert "using B" in code
        assert "struct I" in code
        assert "struct J" in code

    def test_generate_dims_only(self):
        """generate() should work with Dim-only Generators."""
        g = Generators()
        g.X = Dim()
        g.Y = Dim()

        code = generate(g)

        assert "struct X" in code
        assert "struct Y" in code


class TestGeneratorsAttributeAccess:
    """Test attribute access behavior."""

    def test_getattr_raises_for_missing(self):
        """Accessing missing attribute should raise AttributeError."""
        g = Generators()

        with pytest.raises(AttributeError, match="has no attribute 'missing'"):
            _ = g.missing

    def test_contains(self):
        """'in' operator should check registry."""
        g = Generators()
        g.A = Tensor(dtype.float, Dims(i=16))

        assert "A" in g
        assert "B" not in g


class TestGeneratorsPreservesOrder:
    """Test that Generators preserves insertion order."""

    def test_insertion_order(self):
        """Generators should preserve insertion order."""
        g = Generators()
        g.First = Tensor(dtype.float, Dims(i=1))
        g.Second = Tensor(dtype.float, Dims(i=2))
        g.Third = Tensor(dtype.float, Dims(i=3))

        names = list(g.keys())
        assert names == ["First", "Second", "Third"]


class TestCursorWithImplicitDims:
    """Test the CursorInitializer generator."""

    def test_implicit_dim_single(self):
        """implicit_dim with single dimension should generate correct cursor subclass."""
        g = Generators()
        g.AGlobal = Tensor(dtype.half, Dims(warp=4, lane=32, i=8))
        g.ALoadIndex = CompoundIndex(Dims(warp=4, lane=32))
        g.AGlobalLoader = g.AGlobal.initializer(g.ALoadIndex)

        assert g.AGlobalLoader.class_name == "AGlobalLoader"
        assert g.AGlobalLoader.tensor is g.AGlobal

        code = generate(g)
        assert "struct AGlobalLoader : AGlobal::cursor_type" in code
        assert "Base(AGlobal(ptr)[ALoadIndex()])" in code

    def test_implicit_dim_multiple(self):
        """implicit_dim with multiple dimensions should chain subscripts."""
        g = Generators()
        g.AGlobal = Tensor(dtype.half, Dims(warp=4, lane=32, i=8))
        g.WarpIdx = CompoundIndex(Dims(warp=4))
        g.LaneIdx = CompoundIndex(Dims(lane=32))
        g.AGlobalLoader = g.AGlobal.initializer(g.WarpIdx, g.LaneIdx)

        code = generate(g)
        assert "Base(AGlobal(ptr)[WarpIdx()][LaneIdx()])" in code

    def test_implicit_dim_constant_tensor(self):
        """implicit_dim with constant tensor should inherit from cursor with const data."""
        g = Generators()
        g.AGlobal = Tensor(dtype.half, Dims(warp=4, lane=32), constant=True)
        g.ALoadIndex = CompoundIndex(Dims(warp=4, lane=32))
        g.AGlobalLoader = g.AGlobal.initializer(g.ALoadIndex)

        code = generate(g)
        # The struct inherits from the cursor type, which handles const-ness via data_type
        assert "struct AGlobalLoader : AGlobal::cursor_type" in code
        assert "using data_type = typename Base::data_type" in code

    def test_implicit_dim_used_generators(self):
        """implicit_dim's used_generators should include tensor and all dims."""
        g = Generators()
        g.AGlobal = Tensor(dtype.half, Dims(i=8))
        g.Idx1 = CompoundIndex(Dims(j=4))
        g.Idx2 = CompoundIndex(Dims(k=2))
        g.AGlobalLoader = g.AGlobal.initializer(g.Idx1, g.Idx2)

        used = g.AGlobalLoader.used_generators()
        assert g.AGlobal in used
        assert g.Idx1 in used
        assert g.Idx2 in used


class TestStaticDimWithGenerators:
    """Test StaticDim usage with the Generators container."""

    def test_dim_call_in_dims(self):
        """Dim objects can be called to create StaticDim for use in Dims."""
        g = Generators()
        g.I = Dim()
        g.K = Dim()
        g.A = Tensor(dtype.float, Dims(g.I(16), g.K(32)))

        assert g.A.class_name == "A"
        # Access dims to trigger resolution
        dims_dict = dict(g.A.dims.items())
        assert dims_dict == {"I": 16, "K": 32}

    def test_dim_lazy_resolution(self):
        """Dims created before Dim names are set should resolve correctly."""
        g = Generators()
        # Create Dims first, assign to Generators after
        I = Dim()
        K = Dim()
        dims = Dims(I(16), K(32))

        # Now assign to generators - this sets the names
        g.I = I
        g.K = K

        # Names should now be resolvable
        dims_dict = dict(dims.items())
        assert dims_dict == {"I": 16, "K": 32}

    def test_tensor_with_static_dims(self):
        """Tensor with StaticDim-based Dims should generate correct code."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.Matrix = Tensor(dtype.float, Dims(g.I(8), g.J(16)))

        code = generate(g)

        # Should generate dimension structs
        assert "struct I" in code
        assert "struct J" in code
        # Should generate tensor with correct dims
        assert "using Matrix" in code

    def test_tensor_with_lazy_dims(self):
        """Tensor created with unnamed Dims should work after names are set."""
        I = Dim()
        J = Dim()
        tensor = Tensor(dtype.float, Dims(I(4), J(8)))

        g = Generators()
        g.I = I
        g.J = J
        g.MyTensor = tensor

        code = generate(g)
        assert "struct I" in code
        assert "struct J" in code
        assert "using MyTensor" in code

    def test_multiple_tensors_share_dims(self):
        """Multiple tensors can share the same Dim objects."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.K = Dim()
        g.A = Tensor(dtype.float, Dims(g.I(16), g.K(32)))
        g.B = Tensor(dtype.float, Dims(g.K(32), g.J(64)))

        code = generate(g)

        # Dims should only be generated once
        assert code.count("struct I") == 1
        assert code.count("struct J") == 1
        assert code.count("struct K") == 1
        # Both tensors should be generated
        assert "using A" in code
        assert "using B" in code

    def test_compound_index_with_static_dims(self):
        """CompoundIndex should work with StaticDim-based Dims."""
        g = Generators()
        g.Warp = Dim()
        g.Block = Dim()
        g.ThreadIndex = CompoundIndex(Dims(g.Warp(4), g.Block(32)))

        code = generate(g)
        assert "struct WARP" in code
        assert "struct BLOCK" in code
        assert "using ThreadIndex" in code

    def test_tensor_with_static_fold(self):
        """Tensor with StaticFold-based Dims should generate correct code."""
        g = Generators()
        g.K = Dim()
        g.K8 = g.K.fold(8)
        # K8 has stride 8, size 4 -> K (stride 1) has size 8
        g.Matrix = Tensor(dtype.float, Dims(g.K(8), g.K8(4)))

        code = generate(g)

        # Should generate base dimension struct
        assert "struct K" in code
        # Should generate fold alias
        assert "using K8 = spio::Fold<K, 8>" in code
        # Should generate tensor
        assert "using Matrix" in code

    def test_tensor_with_lazy_fold(self):
        """Tensor with unnamed Fold should work after names are set."""
        K = Dim()
        K8 = K.fold(8)
        # K8 has stride 8, size 4 -> K (stride 1) has size 8
        tensor = Tensor(dtype.float, Dims(K(8), K8(4)))

        g = Generators()
        g.K = K
        g.K8 = K8
        g.MyTensor = tensor

        code = generate(g)
        assert "struct K" in code
        assert "using K8 = spio::Fold<K, 8>" in code
        assert "using MyTensor" in code


class TestAnonymousGeneratorsWithStaticDim:
    """Tests for anonymous generators (auto-named) with StaticDim/StaticFold."""

    def test_anonymous_dim_in_tensor(self):
        """Anonymous Dim used in Tensor gets auto-generated name."""
        # Create a Dim without assigning it to Generators
        I = Dim()
        J = Dim()

        # Use in a Tensor
        tensor = Tensor(dtype.float, Dims(I(16), J(32)))

        g = Generators()
        g.MyTensor = tensor  # Only tensor is named, not the dims

        code = generate(g)

        # Both dims should get auto-generated names like _AnonDima, _AnonDimb
        assert "struct _ANONDIM" in code
        assert "using MyTensor" in code

    def test_anonymous_fold_in_tensor(self):
        """Anonymous Fold used in Tensor gets auto-generated name."""
        K = Dim()
        K8 = K.fold(8)

        tensor = Tensor(dtype.float, Dims(K8(4), K(-1)))

        g = Generators()
        g.MyTensor = tensor  # Only tensor is named

        code = generate(g)

        # Dim should get auto-generated name, Fold should reference it
        assert "struct _ANONDIM" in code
        # The fold should use the auto-named dim
        assert "spio::Fold<_ANONDIM" in code
        assert "using MyTensor" in code

    def test_anonymous_fold_with_named_dim(self):
        """Anonymous Fold with a named base Dim generates proper fold alias."""
        I = Dim("I")  # Named dim
        I16 = I.fold(16)  # Anonymous fold (not registered with g)

        tensor = Tensor(dtype.float, Dims(I16(4), I(-1)))

        g = Generators()
        g.MyTensor = tensor

        code = generate(g)

        # I should be a proper Dim class
        assert "struct I : spio::Dim<I>" in code
        # The anonymous fold should generate a Fold alias, not a Dim
        assert "using _ANONFOLD" in code
        assert "spio::Fold<I, 16>" in code
        # Should NOT create a Dim struct for the fold name
        assert "struct _ANONFOLD" not in code

    def test_mixed_named_and_anonymous_dims(self):
        """Mix of named and anonymous dims works correctly."""
        I = Dim()  # Will be anonymous
        J = Dim()  # Will be named

        tensor = Tensor(dtype.float, Dims(I(16), J(32)))

        g = Generators()
        g.J = J  # Only J is named
        g.MyTensor = tensor

        code = generate(g)

        # J should have its explicit name
        assert "struct J" in code
        # I should have an auto-generated name
        assert "struct _ANONDIM" in code
        assert "using MyTensor" in code

    def test_anonymous_dim_shared_across_tensors(self):
        """Anonymous Dim shared by multiple tensors gets single auto-name."""
        K = Dim()  # Anonymous, shared

        tensor1 = Tensor(dtype.float, Dims(K(16)))
        tensor2 = Tensor(dtype.float, Dims(K(32)))

        g = Generators()
        g.A = tensor1
        g.B = tensor2

        code = generate(g)

        # K should appear once with an auto-generated name
        # Count occurrences of "struct _ANONDIM" - should be 1
        struct_count = code.count("struct _ANONDIM")
        assert struct_count == 1, f"Expected 1 struct _ANONDIM, got {struct_count}"

        assert "using A" in code
        assert "using B" in code


class TestDimArgSupport:
    """Tests for classes accepting Dim/Fold objects instead of strings."""

    def test_checkerboard_with_dim_objects(self):
        """Checkerboard accepts Dim objects for dimension args."""
        I = Dim()
        K = Dim()

        g = Generators()
        g.I = I
        g.K = K
        g.cb = Checkerboard(pairs_dim=I, colors_dim=K)

        code = generate(g)

        assert "struct I" in code
        assert "struct K" in code
        assert "using cb = spio::CheckerboardIndex<8, I, K" in code

    def test_checkerboard_with_fold_objects(self):
        """Checkerboard accepts Fold objects for dimension args."""
        I = Dim()
        K8 = I.fold(8)

        g = Generators()
        g.I = I
        g.K8 = K8
        g.cb = Checkerboard(pairs_dim=I, colors_dim=K8)

        code = generate(g)

        assert "struct I" in code
        assert "using K8" in code
        # Fold name becomes uppercase K8
        assert "spio::CheckerboardIndex<8, I, K8" in code or "spio::Fold<" in code

    def test_checkerboard_mixed_str_and_dim(self):
        """Checkerboard accepts mix of strings and Dim objects."""
        I = Dim()

        g = Generators()
        g.I = I
        g.cb = Checkerboard(pairs_dim=I, colors_dim="k8")

        code = generate(g)

        assert "struct I" in code
        assert "spio::Fold<K, 8>" in code  # k8 is parsed as fold

    def test_fragment_with_dim_objects(self):
        """Fragment accepts Dim objects for row/col args."""
        I = Dim()
        K = Dim()

        g = Generators()
        g.I = I
        g.K = K
        g.frag = Fragment(FragmentType.M16_K16_F16_A, row=I, col=K)

        code = generate(g)

        assert "struct I" in code
        assert "struct K" in code
        # Fragment type name includes MMA_ prefix
        assert "MMA_M16_K16_F16_A<I, K>" in code

    def test_fragment_with_fold_objects(self):
        """Fragment accepts Fold objects for row/col args."""
        I = Dim()
        I16 = I.fold(16)

        g = Generators()
        g.frag = Fragment(FragmentType.M16_N16_F32_C, row=I16, col=I)
        g.I = I
        g.I16 = I16

        code = generate(g)

        assert "struct I" in code
        assert "using I16" in code
        # Fragment type name includes MMA_ prefix
        assert "MMA_M16_N16_F32_C<I16, I>" in code

    def test_checkerboard_with_static_dim_auto_size(self):
        """Checkerboard auto-computes size from StaticDim/StaticFold."""
        I = Dim()
        K = Dim()
        K8 = K.fold(8)

        # Using StaticFold objects: i_block/16 gives size=8, K8(4) gives size=4
        # Auto-computed size = 8 * 4 = 32
        i_block = I(128)
        cb = Checkerboard(pairs_dim=i_block / 16, colors_dim=K8(4))

        assert cb.size == 32

    def test_checkerboard_with_static_dim_explicit_size_matches(self):
        """Checkerboard accepts matching explicit size with StaticDim."""
        I = Dim()
        K = Dim()
        K8 = K.fold(8)

        i_block = I(128)
        # Explicit size matches computed: 8 * 4 = 32
        cb = Checkerboard(pairs_dim=i_block / 16, colors_dim=K8(4), size=32)

        assert cb.size == 32

    def test_checkerboard_with_static_dim_mismatched_size_raises(self):
        """Checkerboard raises error if explicit size doesn't match computed."""
        import pytest

        I = Dim()
        K = Dim()
        K8 = K.fold(8)

        i_block = I(128)
        # Computed size would be 8 * 4 = 32, but we specify 64
        with pytest.raises(ValueError, match="doesn't match computed size"):
            Checkerboard(pairs_dim=i_block / 16, colors_dim=K8(4), size=64)

    def test_checkerboard_extracts_dim_from_static(self):
        """Checkerboard extracts underlying Dim/Fold from StaticDim/StaticFold."""
        I = Dim()
        K = Dim()
        K8 = K.fold(8)

        i_block = I(128)
        cb = Checkerboard(pairs_dim=i_block / 16, colors_dim=K8(4))

        # The pairs_dim should be a Fold (from i_block / 16), not StaticFold
        from spio.generators.fold import Fold

        assert isinstance(cb.pairs_dim, Fold)
        assert isinstance(cb.colors_dim, Fold)

    def test_checkerboard_partial_static_uses_default_size(self):
        """Checkerboard with only one Static* arg uses default size."""
        I = Dim()
        K = Dim()
        K8 = K.fold(8)

        # Only colors_dim is a StaticFold, pairs_dim is just a Dim
        cb = Checkerboard(pairs_dim=I, colors_dim=K8(4))

        # Without both having sizes, defaults to 32
        assert cb.size == 32
