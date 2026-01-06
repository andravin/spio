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
    generate,
)


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
