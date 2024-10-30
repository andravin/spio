"""PyTorch module for Conv2d with group width equal to 8."""

from typing import Tuple

import torch
from torch import nn

from ..functional import conv2d_gw8
from ..kernels import Conv2dGw8Params


class Conv2dGw8(nn.Conv2d):
    """PyTorch module for Conv2d with group width equal to 8.

    This module implements 2D convolution with group width equal to 8.
    It is derived from nn.Conv2d and calls the conv2d_gw8 function for
    the forward pass, which in turn calls a custom CUDA kernel.

    The function is designed for channels last memory format and float16
    precision. The weights are maintained in float32 and we use AMP
    (Automatic Mixed Precision) for conversion to float16. Bias is
    optional and will use float32 if present.
    """

    Params = Conv2dGw8Params

    @staticmethod
    def make(*args, **kwargs) -> nn.Module:
        """Create a Conv2dGw8 module if possible.

        If the parameters match the requirements, return a new Conv2dGw8
        module. Otherwise, return None.

        The arguments list is the same as for nn.Conv2d which is also
        the same as Conv2dGw8.match_args(), below.

        Returns None if the parameters do not match the requirements.
        Otherwise, returns a Conv2dGw8 module.
        """
        if Conv2dGw8.match_args(*args, **kwargs):
            return Conv2dGw8(*args, **kwargs)
        return None

    @staticmethod
    def match(module: nn.Module):
        """Check if the requirements for Conv2dGw8 are met."""
        return (
            isinstance(module, nn.Conv2d)
            and not isinstance(module, Conv2dGw8)
            and Conv2dGw8.match_args(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.dilation,
                module.groups,
            )
        )

    @staticmethod
    def match_args(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        **_kwargs,
    ) -> bool:
        """Check if reqirements for Conv2dGw8 are met.

        Returns True if the arguments match the requirements. Otherwise,
        returns False.
        """
        group_width = in_channels // groups
        r, s = _get_pair(kernel_size)
        dy, dx = _get_pair(dilation)
        sy, sx = _get_pair(stride)
        return (
            in_channels == out_channels
            and group_width == 8
            and 1 <= r <= 5
            and 1 <= s <= 5
            and dy == 1
            and dx == 1
            and sy == 1
            and sx == 1
            and padding_mode == "zeros"
        )

    @staticmethod
    def from_torch_module(conv2d: nn.Conv2d):
        """Create a Conv2dGw8 module from a nn.Conv2d module.

        The Conv2dGw8 module will reference the same weight and bias
        tensors as the original Conv2d module. The idea is that you
        replace the original Conv2d module with a new Conv2dGw8 module.
        """
        if not Conv2dGw8.match(conv2d):
            raise ValueError(f"Conv2d {conv2d} does not match Conv2dGw8")
        module = Conv2dGw8(
            in_channels=conv2d.in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size,
            stride=conv2d.stride,
            padding=conv2d.padding,
            groups=conv2d.groups,
            bias=conv2d.bias is not None,
            device=conv2d.weight.device,
        ).to(memory_format=torch.channels_last)
        # Directly assign the weight and bias tensors.
        module.weight = conv2d.weight
        module.bias = conv2d.bias
        return module

    # pylint: disable=arguments-renamed
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Conv2dGw8 module."""
        padding_y, padding_x = _get_pair(self.padding)
        return conv2d_gw8(
            x,
            self.weight,
            self.bias,
            stride=1,
            padding_y=padding_y,
            padding_x=padding_x,
            groups=self.groups,
        )


def _get_pair(x) -> Tuple[int, int]:
    """Return the argument as a pair of integers.

    If the argument is a single integer, return a pair of the same
    integer. Otherwise, return the argument as is.
    """
    if isinstance(x, int):
        return x, x
    return x
