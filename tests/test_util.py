"""Test the utility functions."""

from itertools import product

import torch

from spio.util import divup, next_relative_prime, SixteenChannelsLast


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
    b = SixteenChannelsLast.format(a)
    assert b.shape == (C // 16, K, 16)
    for k, c in product(range(K), range(C)):
        cd16 = c // 16
        cm16 = c % 16
        assert a[k, c] == b[cd16, k, cm16]


def test_sixtenn_channels_last_4d():
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
