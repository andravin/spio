"""Math utility functions."""


def divup(n: int, d: int) -> int:
    """Return the ceiling of n divided by d as an integer."""
    return (n + d - 1) // d
