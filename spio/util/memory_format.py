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

    @staticmethod
    def unformat(tensor: torch.Tensor) -> torch.Tensor:
        """Transpose a tensor from [c16, k, 16c] to [k, c]."""
        if tensor.dim() == 3:
            return tensor.permute(1, 0, 2).contiguous().view(tensor.shape[1], -1)
        elif tensor.dim() == 5:
            _, n, h, w, _ = tensor.shape
            return tensor.permute(1, 0, 4, 2, 3).contiguous().view(n, -1, h, w)
        else:
            raise NotImplementedError(f"{tensor.dim()}D tensor not supported")


def check_channels_last(args: List[torch.Tensor]):
    """Check that the arguments are channels_last tensors."""
    for arg in args:
        if isinstance(arg, torch.Tensor) and len(arg.shape) == 4:
            assert arg.is_contiguous(
                memory_format=torch.channels_last
            ), f"Tensor is not channels_last: {arg}"


class TwoFold:
    """Memory format with two folds.

    Folds M x K to (M / fold_m) x (K / fold_k) x (fold_m) x (fold_k)
    """

    def __init__(self, fold_m: int, fold_k: int):
        self.fold_m = fold_m
        self.fold_k = fold_k

    def format(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fold the given tensor."""
        if tensor.dim() != 2:
            raise NotImplementedError(f"{tensor.dim()}D tensor not supported")
        m, k = tensor.shape
        assert m % self.fold_m == 0, f"m {m} not divisible by fold_m {self.fold_m}"
        assert k % self.fold_k == 0, f"k {k} not divisible by fold_k {self.fold_k}"
        return (
            tensor.view(
                m // self.fold_m,
                self.fold_m,
                k // self.fold_k,
                self.fold_k,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def unformat(self, tensor: torch.Tensor) -> torch.Tensor:
        """Unfold the given tensor."""
        if tensor.dim() != 4:
            raise NotImplementedError(f"{tensor.dim()}D tensor not supported")
        if tensor.shape[2] != self.fold_m or tensor.shape[3] != self.fold_k:
            raise ValueError(
                f"Tensor shape {tensor.shape} does not match fold sizes "
                f"{self.fold_m}, {self.fold_k}"
            )
        return (
            tensor.permute(0, 2, 1, 3)
            .contiguous()
            .view(
                tensor.shape[0] * self.fold_m,
                tensor.shape[1] * self.fold_k,
            )
        )
