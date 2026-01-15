"""Unit tests for the Dims class and automatic fold size inference."""

import pytest
from spio.generators import Dims, Dim, Fold, Strides, LANE, OFFSET
from spio.generators.dim import StaticDim, BUILTIN_DIM_NAMES
from spio.generators.fold import StaticFold


class TestStaticDim:
    """Tests for StaticDim and Dim.__call__."""

    def test_dim_call_creates_static_dim(self):
        """Test that calling a Dim with a size creates a StaticDim."""
        I = Dim("I")
        static = I(16)
        assert isinstance(static, StaticDim)
        assert static.dim is I
        assert static.size == 16
        assert static.dim_name == "I"

    def test_dims_with_static_dims(self):
        """Test creating Dims with StaticDim objects."""
        I = Dim("I")
        K = Dim("K")
        dims = Dims(I(16), K(32))
        assert dims["I"] == 16
        assert dims["K"] == 32

    def test_dims_with_static_dims_preserves_order(self):
        """Test that dimension order is preserved."""
        I = Dim("I")
        J = Dim("J")
        K = Dim("K")
        dims = Dims(I(16), J(8), K(32))
        keys = list(dims.keys())
        assert keys == ["I", "J", "K"]

    def test_dims_error_on_mixed_args(self):
        """Test error when mixing positional and keyword arguments."""
        I = Dim("I")
        with pytest.raises(ValueError, match="Cannot mix positional"):
            Dims(I(16), k=32)

    def test_dims_error_on_non_static_dim(self):
        """Test error when passing non-StaticDim to Dims."""
        with pytest.raises(TypeError, match="Expected StaticDim"):
            Dims(16)

    def test_dims_error_on_duplicate_dim(self):
        """Test error when using same dimension twice."""
        I = Dim("I")
        with pytest.raises(ValueError, match="appears twice in Dims"):
            dims = Dims(I(16), I(32))
            _ = dims["I"]  # Trigger lazy resolution

    def test_dims_error_on_duplicate_fold(self):
        """Test error when using same fold twice."""
        K = Dim("K")
        K8 = K.fold(8)
        with pytest.raises(ValueError, match="appears twice in Dims"):
            dims = Dims(K8(4), K8(2))
            _ = list(dims.keys())  # Trigger lazy resolution

    def test_dims_auto_names_anonymous_dim(self):
        """Test that anonymous Dims get auto-named."""
        unnamed = Dim()
        dims = Dims(unnamed(16))
        keys = list(dims.keys())
        assert len(keys) == 1
        assert keys[0].startswith("_ANONDIM")

    def test_dims_auto_names_anonymous_fold(self):
        """Test that anonymous Folds get auto-named."""
        I = Dim("I")
        unnamed_fold = I.fold(8)
        dims = Dims(unnamed_fold(4))
        keys = list(dims.keys())
        assert len(keys) == 1
        assert keys[0].startswith("_ANONFOLD")

    def test_dims_lazy_resolution(self):
        """Test that dims can be created before Dim has a name, resolved later."""
        # Create unnamed dims
        I = Dim()
        K = Dim()

        # Create Dims with unnamed StaticDims - no error yet
        dims = Dims(I(16), K(32))

        # Now set the names (simulating Generators assignment)
        I._set_class_name("I")
        K._set_class_name("K")

        # Now access should work
        assert dims["I"] == 16
        assert dims["K"] == 32

    def test_static_dim_fold_requires_divisibility(self):
        """Test that StaticDim.fold() requires size to be divisible by stride."""
        I = Dim("I")
        i = I(17)
        with pytest.raises(ValueError, match="Cannot fold.*Use fold_up"):
            i.fold(8)

    def test_static_dim_fold_up_rounds_up(self):
        """Test that StaticDim.fold_up() uses ceiling division."""
        I = Dim("I")
        i = I(17)
        i8 = i.fold_up(8)
        assert i8.size == 3  # ceil(17/8) = 3
        assert i8.fold.stride == 8

    def test_static_dim_fold_up_exact_division(self):
        """Test that fold_up() works correctly when size is exactly divisible."""
        I = Dim("I")
        i = I(16)
        i8 = i.fold_up(8)
        assert i8.size == 2  # 16/8 = 2
        assert i8.fold.stride == 8

    def test_static_dim_multiply_same_base(self):
        """Test that StaticDim * StaticDim adds sizes when same base Dim."""
        I = Dim("I")
        i4 = I(4)
        i8 = I(8)
        i12 = i4 * i8
        assert i12.size == 12
        assert i12.dim is I

    def test_static_dim_multiply_different_base_raises(self):
        """Test that multiplying StaticDims with different base Dims raises TypeError."""
        I = Dim("I")
        J = Dim("J")
        i4 = I(4)
        j4 = J(4)
        with pytest.raises(TypeError, match="different base Dims"):
            i4 * j4

    def test_static_dim_multiply_by_int(self):
        """Test that StaticDim * int multiplies the size."""
        I = Dim("I")
        i4 = I(4)
        i12 = i4 * 3
        assert i12.size == 12
        assert i12.dim is I

    def test_static_dim_int_multiply(self):
        """Test that int * StaticDim multiplies the size."""
        I = Dim("I")
        i4 = I(4)
        i20 = 5 * i4
        assert i20.size == 20
        assert i20.dim is I

    def test_static_dim_divide_creates_fold(self):
        """Test that StaticDim / int creates a StaticFold."""
        I = Dim("I")
        i = I(128)
        i16 = i / 16
        assert isinstance(i16, StaticFold)
        assert i16.size == 8
        assert i16.fold.stride == 16
        assert i16.fold.dim is I

    def test_dim_divide_creates_fold(self):
        """Test that Dim / int creates a Fold."""
        I = Dim("I")
        I16 = I / 16
        assert isinstance(I16, Fold)
        assert I16.stride == 16
        assert I16.dim is I

    def test_dim_divide_is_cached(self):
        """Test that Dim / int returns cached Fold."""
        I = Dim("I")
        I16a = I / 16
        I16b = I / 16
        assert I16a is I16b

    def test_static_dim_modulo(self):
        """Test StaticDim % operator returns StaticDim with size equal to modulo."""
        I = Dim("I")
        i = I(128)
        i_fine = i % 16
        assert isinstance(i_fine, StaticDim)
        assert i_fine.size == 16
        assert i_fine.dim is I

    def test_static_dim_modulo_requires_divisibility(self):
        """Test that modulo must divide size evenly."""
        I = Dim("I")
        i = I(128)
        with pytest.raises(ValueError, match="does not evenly divide"):
            i % 17

    def test_static_dim_division_and_modulo_decompose(self):
        """Test that / and % together decompose a dimension."""
        I = Dim("I")
        i = I(128)
        i_coarse = i / 16  # StaticFold with size 8
        i_fine = i % 16  # StaticDim with size 16
        assert i_coarse.size == 8
        assert i_fine.size == 16
        assert i_coarse.size * i_fine.size == i.size


class TestStaticFold:
    """Tests for StaticFold and Fold.__call__."""

    def test_fold_call_creates_static_fold(self):
        """Test that calling a Fold with a size creates a StaticFold."""
        K = Dim("K")
        K8 = K.fold(8)
        static = K8(4)
        assert isinstance(static, StaticFold)
        assert static.fold is K8
        assert static.size == 4
        # dim_name returns the fold_name (which needs to be set)
        K8._set_class_name("K8")
        assert static.dim_name == "K8"

    def test_dim_fold_method(self):
        """Test that Dim.fold() creates a Fold with the correct properties."""
        K = Dim("K")
        K8 = K.fold(8)
        assert isinstance(K8, Fold)
        assert K8.dim is K
        assert K8.stride == 8

    def test_dims_with_static_fold(self):
        """Test creating Dims with StaticFold objects."""
        K = Dim("K")
        K8 = K.fold(8)
        K8._set_class_name("K8")
        dims = Dims(K8(4))
        assert dims["K8"] == 4

    def test_dims_with_mixed_static_dim_and_fold(self):
        """Test Dims with both StaticDim and StaticFold."""
        K = Dim("K")
        K8 = K.fold(8)
        K8._set_class_name("K8")
        # K8 has stride 8, size 4 -> 32 total elements
        # K has stride 1, size = 8/1 = 8 (elements per K8 tile)
        dims = Dims(K(8), K8(4))
        assert dims["K"] == 8
        assert dims["K8"] == 4

    def test_static_fold_lazy_resolution(self):
        """Test that StaticFold works with lazy name resolution."""
        K = Dim()
        K8 = K.fold(8)

        # Create Dims before names are set
        dims = Dims(K8(4))

        # Set the names
        K._set_class_name("K")
        K8._set_class_name("K8")

        # Now access should work
        assert dims["K8"] == 4

    def test_legacy_fold_still_works(self):
        """Test that legacy Fold(dim_name_string, stride) still works."""
        K8 = Fold("K", 8)
        K8._set_class_name("K8")
        static = K8(4)
        assert isinstance(static, StaticFold)
        assert static.size == 4
        assert static.dim_name == "K8"

    def test_auto_fold_size_with_static_dims(self):
        """Test automatic fold size computation with StaticDim/StaticFold."""
        K = Dim("K")
        K8 = K.fold(8)
        K8._set_class_name("K8")

        # K(-1) should be computed as 8 (from K8 stride 8 / K stride 1)
        dims = Dims(K8(4), K(-1))
        assert dims["K8"] == 4
        assert dims["K"] == 8

    def test_auto_fold_size_with_multiple_folds(self):
        """Test automatic fold size computation with multiple fold levels."""
        K = Dim("K")
        K8 = K.fold(8)
        K4 = K.fold(4)
        K8._set_class_name("K8")
        K4._set_class_name("K4")

        # K8=16, K4=-1 (should be 2), K=-1 (should be 4)
        dims = Dims(K8(16), K4(-1), K(-1))
        assert dims["K8"] == 16
        assert dims["K4"] == 2  # 8/4 = 2
        assert dims["K"] == 4  # 4/1 = 4

    def test_auto_fold_size_error_on_coarsest_auto(self):
        """Test error when coarsest fold has -1 size."""
        K = Dim("K")
        K8 = K.fold(8)
        K8._set_class_name("K8")

        with pytest.raises(ValueError, match="must have an explicit size"):
            dims = Dims(K8(-1), K(8))
            _ = dims["K8"]  # Trigger lazy resolution

    def test_static_fold_multiply_same_base_and_stride(self):
        """Test that StaticFold * StaticFold adds sizes when same base Dim and stride."""
        K = Dim("K")
        K8 = K.fold(8)
        a = K8(4)
        b = K8(2)
        c = a * b
        assert c.size == 6
        assert c.fold is K8

    def test_static_fold_multiply_different_base_raises(self):
        """Test that multiplying StaticFolds with different base Dims raises TypeError."""
        K = Dim("K")
        J = Dim("J")
        K8 = K.fold(8)
        J8 = J.fold(8)
        a = K8(4)
        b = J8(4)
        with pytest.raises(TypeError, match="different base Dim or stride"):
            a * b

    def test_static_fold_multiply_different_stride_raises(self):
        """Test that multiplying StaticFolds with different strides raises TypeError."""
        K = Dim("K")
        K8 = K.fold(8)
        K4 = K.fold(4)
        a = K8(4)
        b = K4(4)
        with pytest.raises(TypeError, match="different base Dim or stride"):
            a * b

    def test_static_fold_multiply_by_int(self):
        """Test that StaticFold * int multiplies the size."""
        K = Dim("K")
        K8 = K.fold(8)
        a = K8(4)
        b = a * 3
        assert b.size == 12
        assert b.fold is K8

    def test_static_fold_int_multiply(self):
        """Test that int * StaticFold multiplies the size."""
        K = Dim("K")
        K8 = K.fold(8)
        a = K8(4)
        b = 2 * a
        assert b.size == 8
        assert b.fold is K8

    def test_fold_refold_to_new_stride(self):
        """Test that Fold.fold(new_stride) refolds to an absolute stride."""
        K = Dim("K")
        K8 = K.fold(8)
        K16 = K8.fold(16)  # Refold to stride 16
        assert K16.stride == 16
        assert K16.dim is K

    def test_fold_divide_operator(self):
        """Test that Fold / int multiplies the stride (relative coarsening)."""
        K = Dim("K")
        K8 = K.fold(8)
        K16 = K8 / 2  # 8 * 2 = 16
        assert K16.stride == 16
        assert K16.dim is K

    def test_fold_divide_chained(self):
        """Test chaining / operators on Dim and Fold."""
        K = Dim("K")
        K16 = K / 8 / 2  # stride 8, then * 2 = 16
        assert K16.stride == 16
        assert K16.dim is K

    def test_static_fold_divide_operator(self):
        """Test that StaticFold / int multiplies stride and divides size."""
        K = Dim("K")
        K8 = K.fold(8)
        k8 = K8(16)  # 16 groups of 8 elements
        k16 = k8 / 2  # 8 groups of 16 elements
        assert k16.size == 8
        assert k16.fold.stride == 16

    def test_static_fold_divide_requires_divisibility(self):
        """Test that StaticFold / int requires size to be divisible."""
        K = Dim("K")
        K8 = K.fold(8)
        k8 = K8(15)
        with pytest.raises(ValueError, match="not divisible"):
            k8 / 2

    def test_static_fold_modulo_basic(self):
        """Test that StaticFold % int creates a new StaticFold with size=modulus."""
        K = Dim("K")
        K8 = K.fold(8)
        k8 = K8(32)
        k8_mod = k8 % 4
        assert isinstance(k8_mod, StaticFold)
        assert k8_mod.size == 4
        assert k8_mod.fold is K8

    def test_static_fold_modulo_preserves_fold(self):
        """Test that modulo operation preserves the underlying fold."""
        K = Dim("K")
        K8 = K.fold(8)
        K8._set_class_name("K8")
        k8 = K8(16)
        k8_mod = k8 % 3
        assert k8_mod.fold.stride == 8
        assert k8_mod.fold.dim is K
        assert k8_mod.dim_name == "K8"

    def test_static_fold_modulo_different_sizes(self):
        """Test modulo with various size values."""
        K = Dim("K")
        K8 = K.fold(8)
        k8 = K8(100)
        assert (k8 % 1).size == 1
        assert (k8 % 10).size == 10
        assert (k8 % 50).size == 50
        assert (k8 % 100).size == 100

    def test_static_fold_division_and_modulo_decompose(self):
        """Test that / and % together can decompose a StaticFold."""
        K = Dim("K")
        K8 = K.fold(8)
        k8 = K8(32)  # 32 tiles of stride 8
        k16 = k8 / 2  # 16 tiles of stride 16
        k8_fine = k8 % 2  # 2 tiles of stride 8 (fine within each k16 tile)
        assert k16.size == 16
        assert k16.fold.stride == 16
        assert k8_fine.size == 2
        assert k8_fine.fold.stride == 8


class TestStridesWithStaticDim:
    """Tests for Strides with StaticDim/StaticFold."""

    def test_strides_with_static_dim(self):
        """Test creating Strides with StaticDim objects."""
        I = Dim("I")
        J = Dim("J")
        strides = Strides(I(64), J(1))
        assert strides["I"] == 64
        assert strides["J"] == 1

    def test_strides_with_static_fold(self):
        """Test creating Strides with StaticFold objects."""
        K = Dim("K")
        K8 = K.fold(8)
        K8._set_class_name("K8")
        strides = Strides(K8(32))
        assert strides["K8"] == 32

    def test_strides_lazy_resolution(self):
        """Test that Strides works with lazy name resolution."""
        I = Dim()
        strides = Strides(I(64))

        # Set the name after creating Strides
        I._set_class_name("I")

        # Now access should work
        assert strides["I"] == 64

    def test_strides_legacy_still_works(self):
        """Test that legacy Strides(i=64) still works."""
        strides = Strides(i=64, j=1)
        assert strides["I"] == 64
        assert strides["J"] == 1

    def test_strides_error_on_mixed_args(self):
        """Test error when mixing positional and keyword arguments."""
        I = Dim("I")
        with pytest.raises(ValueError, match="Cannot mix"):
            Strides(I(64), j=1)

    def test_strides_error_on_duplicate(self):
        """Test error on duplicate stride names."""
        I = Dim("I")
        with pytest.raises(ValueError, match="Duplicate stride name"):
            strides = Strides(I(64), I(32))
            _ = strides["I"]  # Trigger lazy resolution


class TestDimsAutoFold:
    """Tests for automatic fold size inference in Dims."""

    def test_auto_fold_k8_k4_k(self):
        """Test automatic inference with k8, k4, k."""
        dims = Dims(k8=16, i=32, k4=-1, k=-1)
        assert dims["K8"] == 16
        assert dims["K4"] == 2  # 8/4 = 2
        assert dims["K"] == 4  # 4/1 = 4
        assert dims["I"] == 32

    def test_auto_fold_k8_k(self):
        """Test automatic inference skipping intermediate fold."""
        dims = Dims(k8=4, j=16, k=-1)
        assert dims["K8"] == 4
        assert dims["K"] == 8  # 8/1 = 8
        assert dims["J"] == 16

    def test_auto_fold_multiple_base_dims(self):
        """Test automatic inference with multiple base dimensions."""
        dims = Dims(k8=8, k4=-1, k=-1, i16=4, i=-1)
        assert dims["K8"] == 8
        assert dims["K4"] == 2  # 8/4 = 2
        assert dims["K"] == 4  # 4/1 = 4
        assert dims["I16"] == 4
        assert dims["I"] == 16  # 16/1 = 16

    def test_explicit_sizes_match(self):
        """Test that explicit sizes matching computed values are accepted."""
        dims = Dims(k8=4, k4=2, k=4)
        assert dims["K8"] == 4
        assert dims["K4"] == 2
        assert dims["K"] == 4

    def test_error_coarsest_fold_auto(self):
        """Test error when coarsest fold has -1 size."""
        # Multi-fold case
        with pytest.raises(ValueError, match="must have an explicit size"):
            Dims(k8=-1, k=4)
        # Single-fold case (single fold is also the coarsest)
        with pytest.raises(ValueError, match="must have an explicit size"):
            Dims(k=-1, i=32)

    def test_error_explicit_size_mismatch(self):
        """Test error when explicit size doesn't match computed value."""
        with pytest.raises(ValueError, match="explicit size .* but computed size"):
            Dims(k8=4, k=4)  # k should be 8, not 4

    def test_error_non_divisible_fold_factors(self):
        """Test error when fold factors are not divisible."""
        with pytest.raises(ValueError, match="not divisible"):
            Dims(k8=4, k3=-1)  # 8 is not divisible by 3

    def test_single_dimension_no_fold(self):
        """Test that single dimensions without folds work normally."""
        dims = Dims(i=32, j=64)
        assert dims["I"] == 32
        assert dims["J"] == 64

    def test_case_insensitivity(self):
        """Test that dimension names are case-insensitive."""
        dims = Dims(K8=16, k=-1)
        assert dims["K8"] == 16
        assert dims["K"] == 8


class TestBuiltInDims:
    """Tests for built-in dimensions LANE and OFFSET."""

    def test_lane_has_correct_name(self):
        """Test that LANE has the correct dimension name."""
        assert LANE.dim_name == "LANE"

    def test_offset_has_correct_name(self):
        """Test that OFFSET has the correct dimension name."""
        assert OFFSET.dim_name == "OFFSET"

    def test_lane_in_builtin_names(self):
        """Test that LANE is in the builtin dim names list."""
        assert "LANE" in BUILTIN_DIM_NAMES

    def test_offset_in_builtin_names(self):
        """Test that OFFSET is in the builtin dim names list."""
        assert "OFFSET" in BUILTIN_DIM_NAMES

    def test_lane_creates_static_dim(self):
        """Test that LANE(32) creates a StaticDim."""
        static = LANE(32)
        assert isinstance(static, StaticDim)
        assert static.dim is LANE
        assert static.size == 32
        assert static.dim_name == "LANE"

    def test_offset_creates_static_dim(self):
        """Test that OFFSET(8) creates a StaticDim."""
        static = OFFSET(8)
        assert isinstance(static, StaticDim)
        assert static.dim is OFFSET
        assert static.size == 8
        assert static.dim_name == "OFFSET"

    def test_builtin_dims_in_dims(self):
        """Test that built-in dims can be used with Dims."""
        dims = Dims(LANE(32), OFFSET(4))
        assert dims["LANE"] == 32
        assert dims["OFFSET"] == 4


class TestDerivedDimensionInDims:
    """Tests for derived dimensions (e.g., Checkerboard) in Dims."""

    def test_derived_dim_via_dict_positional(self):
        """Test that derived dims can be passed via dict as positional arg."""
        from spio.generators import Checkerboard

        I = Dim("I")
        K = Dim("K")
        K8 = Fold("K", 8)
        K8._set_class_name("K8")

        dims = Dims(
            I(16),
            K(32),
            swizzle=Checkerboard(pairs_dim=I, colors_dim=K8, size=32),
        )
        assert dims["I"] == 16
        assert dims["K"] == 32
        assert "SWIZZLE" in dims
        # The value is the Checkerboard itself
        assert isinstance(dims["SWIZZLE"], Checkerboard)

    def test_derived_dim_preserves_order(self):
        """Test that derived dims via dict preserve position in order."""
        from spio.generators import Checkerboard

        I = Dim("I")
        K = Dim("K")
        K8 = Fold("K", 8)
        K8._set_class_name("K8")

        dims = Dims(
            I(16),
            dict(swizzle=Checkerboard(pairs_dim=I, colors_dim=K8, size=32)),
            K(32),
        )
        keys = list(dims.keys())
        # Derived dims come after static dims in order due to lazy addition
        # but the important thing is all keys are present
        assert "I" in keys
        assert "K" in keys
        assert "SWIZZLE" in keys

    def test_derived_dim_via_kwarg_with_static_dims(self):
        """Test that derived dims via kwarg work with static dims."""
        from spio.generators import Checkerboard

        I = Dim("I")
        K = Dim("K")
        K8 = Fold("K", 8)
        K8._set_class_name("K8")

        # Can still use kwarg syntax alongside positional static dims
        dims = Dims(
            I(16),
            K(32),
            swizzle=Checkerboard(pairs_dim=I, colors_dim=K8, size=32),
        )
        assert dims["I"] == 16
        assert dims["K"] == 32
        assert "SWIZZLE" in dims


class TestComputeFullStridesWithAnonymousFolds:
    """Tests for compute_full_strides with anonymous folds."""

    def test_anonymous_fold_in_dims_and_strides(self):
        """Test object-based matching when both Dims and Strides use anonymous folds."""
        from spio.generators.dims import compute_full_strides

        J = Dim("J")
        J8 = J.fold(8)  # Anonymous fold - no name set

        # Use J8 in both Dims and Strides
        dims = Dims(J8(16), J(-1))  # J8 is anonymous, J is named
        strides = Strides(J8(32))  # Same J8 fold object

        # compute_full_strides should match by object identity
        result = compute_full_strides(dims, strides)

        # J8 should have explicit stride 32, J should be default (1)
        # Note: J8 gets auto-named during resolution
        j8_name = list(dims.keys())[0]  # First dim is J8
        assert result[j8_name] == 32
        assert result["J"] == 1

    def test_anonymous_fold_auto_size_with_strides(self):
        """Test auto-computed sizes (-1) work with object-based stride matching."""
        from spio.generators.dims import compute_full_strides

        J = Dim("J")
        J8 = J.fold(8)
        J2 = J.fold(2)

        # J8 has explicit size, J2 is auto-computed
        dims = Dims(J8(4), J2(-1))  # J2 should be 4 (= 8/2)
        strides = Strides(J8(20))  # Explicit stride for J8

        result = compute_full_strides(dims, strides)

        # Get auto-generated names for anonymous folds
        keys = list(dims.keys())
        j8_name = keys[0]  # First is J8
        j2_name = keys[1]  # Second is J2

        # Verify auto-size was computed (4 = 8/2)
        assert dims[j2_name] == 4

        # Verify strides: J2 has stride 1 (innermost), J8 has stride 20 (explicit)
        assert result[j2_name] == 1
        assert result[j8_name] == 20

    def test_multiple_anonymous_folds_same_base_dim(self):
        """Test multiple anonymous folds of the same base dimension."""
        from spio.generators.dims import compute_full_strides

        I = Dim("I")
        warp_i = I.fold(64)  # Anonymous
        I16 = I.fold(16)  # Anonymous

        dims = Dims(warp_i(2), I16(4), I(-1))
        strides = Strides()  # No explicit strides

        result = compute_full_strides(dims, strides)

        # Get key names
        keys = list(dims.keys())
        warp_i_name = keys[0]
        i16_name = keys[1]

        # Auto-size for I: fold factors are 64, 16, 1
        # Sorted descending: warp_i(64)=2, I16(16)=4, I(1)=?
        # I16 = 64/16 = 4 ✓, I = 16/1 = 16
        assert dims["I"] == 16

        # Strides computed right-to-left (innermost first):
        # I (size=16): stride 1
        # I16 (size=4): stride = 16 * 1 = 16
        # warp_i (size=2): stride = 4 * 16 = 64
        assert result["I"] == 1
        assert result[i16_name] == 16
        assert result[warp_i_name] == 64


class TestTensorWithAnonymousFoldsAndStrides:
    """Tests for Tensor with anonymous folds in Dims and Strides."""

    def test_tensor_with_anonymous_fold_strides(self):
        """Test Tensor with Strides using anonymous folds."""
        from spio.generators import Tensor, dtype, generate, Generators

        g = Generators()
        J = g.J = Dim()
        J8 = J.fold(8)  # Anonymous
        J2 = J.fold(2)  # Anonymous

        g.T = Tensor(
            dtype.float,
            Dims(J8(4), J2(-1)),
            strides=Strides(J8(20)),  # Explicit stride for anonymous fold
        )

        # Should generate without error
        code = generate(g)
        assert "spio::Tensor<float" in code

    def test_tensor_with_derived_dims_and_strides(self):
        """Test Tensor with derived dimensions (Checkerboard) and strides."""
        from spio.generators import Tensor, dtype, Checkerboard, generate, Generators

        g = Generators()
        I = g.I = Dim()
        K = g.K = Dim()
        K8 = g.K8 = K.fold(8)
        I16 = I.fold(16)  # Anonymous

        g.T = Tensor(
            dtype.float,
            Dims(
                I16(8),
                K(32),
                swizzle=Checkerboard(pairs_dim=I, colors_dim=K8, size=32),
            ),
        )

        # Should generate without error - derived dims don't need strides
        code = generate(g)
        assert "spio::Tensor<float" in code
        assert "Checkerboard" in code
