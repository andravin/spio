"""Helper functions for dealing with CUDA architectures."""

from typing import Tuple


def sm_from_arch(arch: Tuple[int, int]) -> str:
    """Return the corresponding sm_?? string for an architecture tuple."""
    if isinstance(arch, tuple):
        return f"sm_{arch[0]}{arch[1]}"
    else:
        return arch
