"""Define the parameters class for the Conv2dGw8 kernel."""

from dataclasses import dataclass
import re
from typing import Tuple, Union

import torch


@dataclass(frozen=True)
class Conv2dGw8Params:
    """Parameters for a 2D convolution with group width 8.

    This class is used to validate and store parameters for a grouped convolution operation.

    Attributes:
        N (int): Batch size.
        C (int): Number of input channels.
        H (int): Height of the input.
        W (int): Width of the input.
        padding (int or tuple): Padding for height and width.
        R (int): Height of the convolution kernel.
        S (int): Width of the convolution kernel.
        has_bias (bool): Whether the convolution has a bias term (default is False).
    """

    n: int
    c: int
    h: int
    w: int
    padding: Union[int, Tuple[int, int]] = 1  # Also allows tuple (padding_h, padding_w)
    r: int = 3
    s: int = 3
    has_bias: bool = False
    group_width: int = 8
    stride: int = 1

    def encode(self) -> str:
        """Return a string representation of the parameters."""
        bias_str = "b" if self.has_bias else "nb"
        return (
            f"{self.n}n_{self.c}c_{self.h}h_{self.w}w_{self.padding_h}py_"
            f"{self.padding_w}px_{self.r}r_{self.s}s_{bias_str}"
        )

    @classmethod
    def decode(cls, string: str) -> "Conv2dGw8Params":
        """Get a Conv2dGw8Params instance from a string rep."""
        matches = re.match(
            r"(\d+)n_(\d+)c_(\d+)h_(\d+)w_(\d+)py_(\d+)px_(\d+)r_(\d+)s_(b|nb)", string
        )
        if matches is None:
            raise ValueError("Invalid string representation.")
        groups = matches.groups()
        n, c, h, w, padding_h, padding_w, r, s = map(int, groups[:8])
        has_bias = groups[8] == "b"
        return cls(
            n=n,
            c=c,
            h=h,
            w=w,
            padding=(padding_h, padding_w),
            r=r,
            s=s,
            has_bias=has_bias,
        )

    @staticmethod
    def from_tensors(inputs, weight, bias, padding=1):
        """Derive a Conv2dGw8Params instance from tensor args.

        The tensors are arguments from a torch.nn.functional.conv2d call.

        Args:
            input (torch.Tensor): The input tensor.
            weight (torch.Tensor): The weight tensor.
            bias (torch.Tensor or None): The bias tensor or None.
            padding (int or tuple): Padding for height and width.
        """
        assert bias is None or (
            len(bias.shape) == 1 and bias.shape[0] == weight.shape[0]
        )
        n, c, h, w = inputs.shape
        k, group_width, r, s = weight.shape
        assert (
            group_width == Conv2dGw8Params.group_width
        ), "Only group width of 8 is supported."
        assert k == c, "Number of output channels must match number of input channels."
        has_bias = bias is not None
        params = Conv2dGw8Params(
            n=n,
            c=c,
            h=h,
            w=w,
            r=r,
            s=s,
            padding=padding,
            has_bias=has_bias,
        )
        params.validate()
        return params

    @staticmethod
    def from_torch_module(module, inputs: torch.Tensor):
        """Derive a Conv2dGw8Params instance from a torch.nn.Module."""
        n, c, h, w = inputs.shape
        k, _group_width, r, s = module.weight.shape
        assert (
            k == c
        ), f"Number of output channels must match number of input channels. {k} != {c}"
        has_bias = module.bias is not None
        padding = module.padding
        return Conv2dGw8Params(
            n=n,
            c=c,
            h=h,
            w=w,
            r=r,
            s=s,
            padding=padding,
            has_bias=has_bias,
        )

    def is_valid(self) -> bool:
        """Return True if the parameters are valid, otherwise False."""
        try:
            self.validate()
            return True
        except AssertionError:
            return False

    def validate(self):
        """Assert that the parameters are valid."""
        assert self.n > 0
        assert self.c > 0
        assert self.h > 0
        assert self.w > 0
        assert self.padding_h >= 0
        assert self.padding_w >= 0
        assert (
            self.c % self.group_width == 0
        ), "Number of channels must be divisible by group width."
        assert self.r in range(1, 6), "Kernel height must be between 1 and 5."
        assert self.s in range(1, 6), "Kernel width must be between 1 and 5."

    @property
    def groups(self):
        """Return the number of groups."""
        return self.c // self.group_width

    @property
    def padding_h(self):
        """Return the height padding."""
        return self.padding[0] if isinstance(self.padding, tuple) else self.padding

    @property
    def padding_w(self):
        """Return the width padding."""
        return self.padding[1] if isinstance(self.padding, tuple) else self.padding

    @property
    def transpose_padding_h(self):
        """Return the height padding of the transposed convolution."""
        return self.r - 1 - self.padding_h

    @property
    def transpose_padding_w(self):
        """Return the width padding of the transposed convolution."""
        return self.s - 1 - self.padding_w

    @property
    def k(self):
        """Return the number of output channels.

        Equal to the number of input channels.
        """
        return self.c

    @property
    def p(self):
        """Return the height of the output."""
        return self.h + 2 * self.padding_h - self.r + 1

    @property
    def q(self):
        """Return the width of the output."""
        return self.w + 2 * self.padding_w - self.s + 1

    @property
    def kernel_size(self):
        """Return the kernel size as a tuple (R, S)."""
        return (self.r, self.s)

    @property
    def input_shape(self):
        """Return the shape of the input tensor as a tuple (N, C, H,
        W)."""
        return (self.n, self.c, self.h, self.w)

    @property
    def output_shape(self):
        """Return the shape of the output tensor as a tuple (N, C, P,
        Q)."""
        return (self.n, self.k, self.p, self.q)

    @property
    def weight_shape(self):
        """Return the shape of the weight tensor as a tuple (K,
        group_width, R, S)."""
        return (self.k, self.group_width, self.r, self.s)

    @property
    def bias_shape(self):
        """Return the shape of the bias tensor as a tuple (K,)."""
        return (self.k,)
