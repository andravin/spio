"""Helper functions for dealing with CUDA architectures."""

from typing import Tuple


def sm_from_arch(arch: Tuple[int, int]) -> str:
    """Return a sm_xx string for an arch tuple."""
    if isinstance(arch, tuple):
        return f"sm_{arch[0]}{arch[1]}"
    return arch
