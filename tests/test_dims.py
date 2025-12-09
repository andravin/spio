"""Unit tests for the Dims class and automatic fold size inference."""

import pytest
from spio.generators import Dims


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
