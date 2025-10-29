"""Utility functions for converting tensor formats."""

from typing import List, Any

import torch


def to_channels_last(*args) -> List[Any]:
    """Convert all 4-D tensors to channels_last memory format."""
    return tuple(
        (
            t.contiguous(memory_format=torch.channels_last)
            if isinstance(t, torch.Tensor) and len(t.shape) == 4
            else t
        )
        for t in args
    )
