"""Math utility functions."""

import math


def divup(n: int, d: int) -> int:
    """Return the ceiling of n divided by d as an integer."""
    return (n + d - 1) // d


def next_relative_prime(a: int, b: int) -> int:
    """Return the least number r > = a such that r and b are relative prime.

    Args:
        a: The number to start the search from.
        b: The number to be relative prime with.
    """
    for r in range(a, a * b):
        if math.gcd(r, b) == 1:
            return r
    raise AssertionError(f"Could not find relative prime for {a} and {b}.")
