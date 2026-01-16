"""Unit tests for Fragment and related classes."""

import pytest

from spio.generators import (
    Fragment,
    FragmentType,
    Dim,
    Generators,
    generate,
    dtype,
)


class TestFragmentSpecs:
    """Tests for _FragmentSpecs dataclass and FragmentType enum."""

    def test_fragment_type_has_class_name(self):
        """FragmentType values should have class_name attribute."""
        assert FragmentType.M16_N8_F32_C.value.class_name == "MMA_M16_N8_F32_C"
        assert FragmentType.M16_N16_F32_C.value.class_name == "MMA_M16_N16_F32_C"
        assert FragmentType.M16_K8_F16_A.value.class_name == "MMA_M16_K8_F16_A"

    def test_fragment_type_has_num_rows(self):
        """FragmentType values should have num_rows attribute."""
        assert FragmentType.M16_N8_F32_C.value.num_rows == 16
        assert FragmentType.M16_N16_F32_C.value.num_rows == 16
        assert FragmentType.N8_K8_F16_B.value.num_rows == 8
        assert FragmentType.N16_K16_F16_B.value.num_rows == 16

    def test_fragment_type_has_num_cols(self):
        """FragmentType values should have num_cols attribute."""
        assert FragmentType.M16_N8_F32_C.value.num_cols == 8
        assert FragmentType.M16_N16_F32_C.value.num_cols == 16
        assert FragmentType.N8_K8_F16_B.value.num_cols == 8
        assert FragmentType.N8_K16_F16_B.value.num_cols == 16

    def test_fragment_type_has_dtype(self):
        """FragmentType values should have dtype attribute."""
        assert FragmentType.M16_N8_F32_C.value.dtype == dtype.float
        assert FragmentType.M16_N16_F32_C.value.dtype == dtype.float
        assert FragmentType.M16_K8_F16_A.value.dtype == dtype.half
        assert FragmentType.M16_K16_F16_A.value.dtype == dtype.half
        assert FragmentType.N8_K8_F16_B.value.dtype == dtype.half


class TestFragmentSize:
    """Tests for Fragment.size property and _FragmentSize class."""

    def test_size_returns_row_size_for_row_dim(self):
        """Fragment.size(row_dim) should return num_rows."""
        I = Dim("I")
        K = Dim("K")
        frag = Fragment(FragmentType.M16_K16_F16_A, row=I, col=K)

        assert frag.size(I) == 16

    def test_size_returns_col_size_for_col_dim(self):
        """Fragment.size(col_dim) should return num_cols."""
        I = Dim("I")
        K = Dim("K")
        frag = Fragment(FragmentType.M16_K16_F16_A, row=I, col=K)

        assert frag.size(K) == 16

    def test_size_with_string_dims(self):
        """Fragment.size should work when fragment uses string dims."""
        frag = Fragment(FragmentType.M16_N8_F32_C, row="m", col="n")

        assert frag.size("m") == 16
        assert frag.size("M") == 16  # Case-insensitive
        assert frag.size("n") == 8
        assert frag.size("N") == 8

    def test_size_with_mixed_dim_types(self):
        """Fragment.size should work with Dim objects when fragment uses strings."""
        I = Dim("I")
        frag = Fragment(FragmentType.M16_N8_F32_C, row="i", col="n")

        # Should be able to query with Dim object that has matching name
        assert frag.size(I) == 16

    def test_size_with_fold_dims(self):
        """Fragment.size should work with Fold objects."""
        g = Generators()
        g.I = Dim()
        g.I16 = g.I.fold(16)
        g.K = Dim()

        frag = Fragment(FragmentType.M16_K8_F16_A, row=g.I16, col=g.K)

        assert frag.size(g.I16) == 16
        assert frag.size(g.K) == 8

    def test_size_raises_for_unknown_dim(self):
        """Fragment.size should raise ValueError for unknown dimension."""
        I = Dim("I")
        K = Dim("K")
        J = Dim("J")
        frag = Fragment(FragmentType.M16_K16_F16_A, row=I, col=K)

        with pytest.raises(ValueError, match="not part of fragment"):
            frag.size(J)

    def test_size_raises_for_unknown_string_dim(self):
        """Fragment.size should raise ValueError for unknown string dimension."""
        frag = Fragment(FragmentType.M16_N8_F32_C, row="m", col="n")

        with pytest.raises(ValueError, match="not part of fragment"):
            frag.size("k")

    def test_size_different_fragment_types(self):
        """Fragment.size should return correct sizes for different fragment types."""
        I = Dim("I")
        J = Dim("J")
        K = Dim("K")

        # Test accumulator fragment
        c_frag = Fragment(FragmentType.M16_N16_F32_C, row=I, col=J)
        assert c_frag.size(I) == 16
        assert c_frag.size(J) == 16

        # Test A fragment
        a_frag = Fragment(FragmentType.M16_K8_F16_A, row=I, col=K)
        assert a_frag.size(I) == 16
        assert a_frag.size(K) == 8

        # Test B fragment
        b_frag = Fragment(FragmentType.N8_K16_F16_B, row=J, col=K)
        assert b_frag.size(J) == 8
        assert b_frag.size(K) == 16


class TestFragmentBasic:
    """Tests for basic Fragment functionality."""

    def test_fragment_with_string_dims(self):
        """Fragment should accept string dimension names."""
        frag = Fragment(FragmentType.M16_N8_F32_C, row="m", col="n", class_name="Acc")

        assert frag.class_name == "Acc"
        assert frag.dim_names == ("M", "N")

    def test_fragment_with_dim_objects(self):
        """Fragment should accept Dim objects."""
        I = Dim("I")
        J = Dim("J")
        frag = Fragment(FragmentType.M16_N8_F32_C, row=I, col=J, class_name="Acc")

        assert frag.dim_names == ("I", "J")

    def test_fragment_generates_type_alias(self):
        """Fragment.generate() should create a type alias."""
        frag = Fragment(FragmentType.M16_N8_F32_C, row="m", col="n", class_name="Acc")

        code = frag.generate()

        assert "using Acc = spio::MMA_M16_N8_F32_C<M, N>;" in code

    def test_fragment_with_generators(self):
        """Fragment should integrate with Generators container."""
        g = Generators()
        g.I = Dim()
        g.J = Dim()
        g.CFragment = Fragment(FragmentType.M16_N16_F32_C, row=g.I, col=g.J)

        assert g.CFragment.class_name == "CFragment"

        code = generate(g)
        assert "struct I" in code
        assert "struct J" in code
        assert "using CFragment = spio::MMA_M16_N16_F32_C<I, J>" in code


class TestFragmentLoadIndex:
    """Tests for FragmentLoadIndex."""

    def test_load_index_class_name(self):
        """FragmentLoadIndex should have correct class name."""
        frag = Fragment(
            FragmentType.M16_K16_F16_A, row="m", col="k", class_name="AFrag"
        )

        load_idx = frag.load_index

        assert load_idx.get_class_name() == "AFrag::load_index_type"

    def test_load_index_generates_nothing(self):
        """FragmentLoadIndex.generate() should return empty string."""
        frag = Fragment(
            FragmentType.M16_K16_F16_A, row="m", col="k", class_name="AFrag"
        )

        load_idx = frag.load_index

        assert load_idx.generate() == ""


class TestFragmentCompoundIndex:
    """Tests for FragmentCompoundIndex."""

    def test_compound_index_class_name(self):
        """FragmentCompoundIndex should have correct class name."""
        frag = Fragment(FragmentType.M16_N8_F32_C, row="m", col="n", class_name="Acc")

        compound_idx = frag.compound_index

        assert compound_idx.get_class_name() == "Acc::compound_index_type"

    def test_compound_index_generates_nothing(self):
        """FragmentCompoundIndex.generate() should return empty string."""
        frag = Fragment(FragmentType.M16_N8_F32_C, row="m", col="n", class_name="Acc")

        compound_idx = frag.compound_index

        assert compound_idx.generate() == ""


class TestTensorDivideByFragment:
    """Tests for Tensor / Fragment division operator."""

    def test_basic_division_with_string_dims(self):
        """Tensor / Fragment should divide matching dimensions."""
        from spio.generators import Tensor, Dims

        tensor = Tensor(dtype.half, Dims(m=64, k=32))
        frag = Fragment(
            FragmentType.M16_K16_F16_A, row="m", col="k", class_name="AFrag"
        )

        result = tensor / frag

        # M dimension divided by 16, K dimension divided by 16
        assert dict(result.dims.items()) == {"M": 4, "K": 2}
        assert result.data_type is frag

    def test_division_with_dim_objects(self):
        """Tensor / Fragment should work with Dim objects and produce folds."""
        from spio.generators import Tensor, Dims
        from spio.generators.fold import StaticFold

        g = Generators()
        g.I = Dim()
        g.K = Dim()

        tensor = Tensor(dtype.half, Dims(g.I(64), g.K(32)))
        frag = Fragment(
            FragmentType.M16_K16_F16_A, row=g.I, col=g.K, class_name="AFrag"
        )

        result = tensor / frag

        # The result dims should use the divided StaticFold objects
        # Check by iterating over the dim args
        dim_args = list(result.dims.iter_args())
        assert len(dim_args) == 2

        # Both should be StaticFolds now (divided from StaticDims)
        assert isinstance(dim_args[0], StaticFold)
        assert isinstance(dim_args[1], StaticFold)

        # Check sizes: 64/16 = 4 and 32/16 = 2
        assert dim_args[0].size == 4
        assert dim_args[1].size == 2

    def test_division_produces_static_folds(self):
        """Tensor / Fragment should produce StaticFolds from StaticDims."""
        from spio.generators import Tensor, Dims
        from spio.generators.fold import StaticFold
        from spio.generators.dim import StaticDim

        g = Generators()
        g.I = Dim()
        g.K = Dim()

        # Create tensor with StaticDims
        i_warp = g.I(64)
        k_chunk = g.K(32)
        tensor = Tensor((k_chunk, i_warp), data_type=dtype.half)

        frag = Fragment(
            FragmentType.M16_K16_F16_A, row=g.I, col=g.K, class_name="AFrag"
        )

        result = tensor / frag

        # After division, both dimensions should be StaticFolds
        dim_args = list(result.dims.iter_args())
        assert all(isinstance(arg, StaticFold) for arg in dim_args)

        # The folds should have the correct stride (16) from the fragment size
        # k_chunk/16 -> fold with stride 16, size 2
        # i_warp/16 -> fold with stride 16, size 4
        for arg in dim_args:
            assert arg.fold.stride == 16

    def test_division_sets_data_type_to_fragment(self):
        """Tensor / Fragment should set data_type to the Fragment."""
        from spio.generators import Tensor, Dims

        tensor = Tensor(dtype.half, Dims(m=32, k=16))
        frag = Fragment(
            FragmentType.M16_K16_F16_A, row="m", col="k", class_name="AFrag"
        )

        result = tensor / frag

        assert result.data_type is frag

    def test_division_preserves_constant(self):
        """Tensor / Fragment should preserve the constant attribute."""
        from spio.generators import Tensor, Dims

        tensor = Tensor(dtype.half, Dims(m=32, k=16), constant=True)
        frag = Fragment(
            FragmentType.M16_K16_F16_A, row="m", col="k", class_name="AFrag"
        )

        result = tensor / frag

        assert result.constant is True

    def test_division_with_extra_tensor_dims(self):
        """Tensor / Fragment should preserve non-matching dimensions."""
        from spio.generators import Tensor, Dims

        # Tensor has an extra dimension 'n' not in the fragment
        tensor = Tensor(dtype.half, Dims(m=64, k=32, n=4))
        frag = Fragment(
            FragmentType.M16_K16_F16_A, row="m", col="k", class_name="AFrag"
        )

        result = tensor / frag

        result_dims = dict(result.dims.items())
        assert result_dims["M"] == 4  # 64 / 16
        assert result_dims["K"] == 2  # 32 / 16
        assert result_dims["N"] == 4  # unchanged

    def test_division_dtype_mismatch_raises(self):
        """Tensor / Fragment should raise if dtypes don't match."""
        from spio.generators import Tensor, Dims

        # Tensor has float dtype, Fragment expects half
        tensor = Tensor(dtype.float, Dims(m=32, k=16))
        frag = Fragment(
            FragmentType.M16_K16_F16_A, row="m", col="k", class_name="AFrag"
        )

        with pytest.raises(ValueError, match="dtype.*doesn't match"):
            tensor / frag

    def test_division_missing_dim_raises(self):
        """Tensor / Fragment should raise if fragment dim not in tensor."""
        from spio.generators import Tensor, Dims

        # Tensor is missing the 'k' dimension
        tensor = Tensor(dtype.half, Dims(m=32, n=16))
        frag = Fragment(
            FragmentType.M16_K16_F16_A, row="m", col="k", class_name="AFrag"
        )

        with pytest.raises(ValueError, match="not found in Tensor"):
            tensor / frag

    def test_division_not_divisible_raises(self):
        """Tensor / Fragment should raise if dimension not evenly divisible."""
        from spio.generators import Tensor, Dims

        # M dimension (30) not divisible by fragment row size (16)
        tensor = Tensor(dtype.half, Dims(m=30, k=32))
        frag = Fragment(
            FragmentType.M16_K16_F16_A, row="m", col="k", class_name="AFrag"
        )

        with pytest.raises(ValueError, match="not.*evenly divisible"):
            tensor / frag

    def test_division_with_non_fragment_raises(self):
        """Tensor / non-Fragment should raise TypeError."""
        from spio.generators import Tensor, Dims

        tensor = Tensor(dtype.half, Dims(m=32, k=16))

        with pytest.raises(TypeError, match="requires a Fragment"):
            tensor / 16

    def test_division_case_insensitive(self):
        """Tensor / Fragment dimension matching should be case-insensitive."""
        from spio.generators import Tensor, Dims

        # Tensor uses lowercase, Fragment uses uppercase (or vice versa)
        tensor = Tensor(dtype.half, Dims(M=64, K=32))  # uppercase
        frag = Fragment(
            FragmentType.M16_K16_F16_A, row="m", col="k", class_name="AFrag"
        )  # lowercase

        result = tensor / frag

        result_dims = dict(result.dims.items())
        assert result_dims["M"] == 4
        assert result_dims["K"] == 2

    def test_division_different_fragment_sizes(self):
        """Tensor / Fragment should handle fragments with different row/col sizes."""
        from spio.generators import Tensor, Dims

        # M16_N8_F32_C has 16 rows, 8 cols
        tensor = Tensor(dtype.float, Dims(m=32, n=16))
        frag = Fragment(FragmentType.M16_N8_F32_C, row="m", col="n", class_name="CFrag")

        result = tensor / frag

        result_dims = dict(result.dims.items())
        assert result_dims["M"] == 2  # 32 / 16
        assert result_dims["N"] == 2  # 16 / 8
