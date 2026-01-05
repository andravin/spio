"""Test the utility functions."""

from itertools import product

import torch

from spio.util import divup, next_relative_prime, SixteenChannelsLast, TwoFold


def test_divup():
    """Test the divup function."""
    assert divup(10, 3) == 4


def test_next_relative_prime():
    """Test the get_conflict_free_spacing function."""
    for n in range(1, 20):
        assert next_relative_prime(8 * n, 8) == 8 * n + 1
    assert next_relative_prime(1, 8) == 1
    assert next_relative_prime(2, 8) == 3
    assert next_relative_prime(3, 8) == 3
    assert next_relative_prime(4, 8) == 5
    assert next_relative_prime(5, 8) == 5
    assert next_relative_prime(6, 8) == 7
    assert next_relative_prime(7, 8) == 7


def test_sixteen_channels_last_2d():
    """Test the SixteenChannelsLast memory format with 2d tensors."""
    K, C = (256, 64)
    a = torch.randn(K, C)
    a_16c = SixteenChannelsLast.format(a)
    assert a_16c.shape == (C // 16, K, 16)
    for k, c in product(range(K), range(C)):
        cd16 = c // 16
        cm16 = c % 16
        assert a[k, c] == a_16c[cd16, k, cm16]

    b = SixteenChannelsLast.unformat(a_16c)
    assert torch.equal(a, b)


def test_sixteen_channels_last_4d():
    """Test the SixteenChannelsLast memory format with 4d tensors."""
    N, C, H, W = (2, 64, 8, 16)
    for a_format in [torch.contiguous_format, torch.channels_last]:
        a = torch.randn(N, C, H, W).to(memory_format=a_format)
        b = SixteenChannelsLast.format(a)
        assert b.shape == (C // 16, N, H, W, 16)
        for n, c, h, w in product(range(N), range(C), range(H), range(W)):
            cd16 = c // 16
            cm16 = c % 16
            assert a[n, c, h, w] == b[cd16, n, h, w, cm16]


def test_two_fold():
    """Test the TwoFold memory format."""
    fold_m = 4
    fold_k = 8
    M, K = 16, 32
    a = torch.randn(M, K)
    two_fold = TwoFold(fold_m, fold_k)
    a_folded = two_fold.format(a)
    assert a_folded.shape == (M // fold_m, K // fold_k, fold_m, fold_k)
    for m, k in product(range(M), range(K)):
        md = m // fold_m
        mm = m % fold_m
        kd = k // fold_k
        km = k % fold_k
        assert a[m, k] == a_folded[md, kd, mm, km]

    b = two_fold.unformat(a_folded)
    assert torch.equal(a, b)


def test_two_fold_unformat():
    """Test the TwoFold unformat error handling."""
    fold_m = 4
    fold_k = 8

    M = 2 * fold_m
    K = 3 * fold_k

    # Test with incorrect shape
    a_folded = torch.randn(2, 3, fold_m, fold_k)  # Invalid shape

    two_fold = TwoFold(fold_m, fold_k)
    a = two_fold.unformat(a_folded)
    assert a.shape == (M, K)
    for m, k in product(range(M), range(K)):
        md = m // fold_m
        mm = m % fold_m
        kd = k // fold_k
        km = k % fold_k
        assert a[m, k] == a_folded[md, kd, mm, km]
