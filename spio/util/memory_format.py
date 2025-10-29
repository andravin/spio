"""Implement custom memory formats for PyTorch tensors."""

import torch

from typing import List


class SixteenChannelsLast:
    """Memory format for 16 channels last."""

    @staticmethod
    def format(tensor: torch.Tensor) -> torch.Tensor:
        """Transpose a tensor from [k, c] to [c16, k, 16c]."""
        if tensor.dim() == 2:
            return tensor.view(tensor.shape[0], -1, 16).permute(1, 0, 2).contiguous()
        elif tensor.dim() == 4:
            n, _, h, w = tensor.shape
            return tensor.view(n, -1, 16, h, w).permute(1, 0, 3, 4, 2).contiguous()
        else:
            raise NotImplementedError(f"{tensor.dim()}D tensor not supported")


def check_channels_last(args: List[torch.Tensor]):
    """Check that the arguments are channels_last tensors."""
    for arg in args:
        if isinstance(arg, torch.Tensor) and len(arg.shape) == 4:
            assert arg.is_contiguous(
                memory_format=torch.channels_last
            ), f"Tensor is not channels_last: {arg}"
