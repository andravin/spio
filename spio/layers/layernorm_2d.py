"""PyTorch module that implements LayerNorm2d."""

import torch
from torch import nn

from ..functional import layernorm_2d


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm2d module."""

    @staticmethod
    def make(*args, **kwargs) -> nn.Module:
        """Create a LayerNorm2d module for the given arguments.

        Returns a LayerNorm2d module if the arguments match the requirements.
        Otherwise, returns None.
        """
        if LayerNorm2d.match_args(*args, **kwargs):
            return LayerNorm2d(*args, **kwargs)
        return None

    @staticmethod
    def match_args(normalized_shape, **_kwargs):
        """Check if the arguments match the requirements for LayerNorm2d.

        Returns true if the arguments match the requirements.
        """
        if not isinstance(normalized_shape, int):
            if len(normalized_shape) != 1:
                return False
            channels = normalized_shape[0]
        else:
            channels = normalized_shape
        return channels % 8 == 0 and channels <= 2048

    # pylint: disable=arguments-renamed
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return layernorm_2d(x, weight=self.weight, bias=self.bias, eps=self.eps)
