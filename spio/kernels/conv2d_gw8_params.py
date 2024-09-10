from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Conv2dGw8Params:
    """A dataclass that holds parameters for a 2D convolution with group width 8.

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

    N: int
    C: int
    H: int
    W: int
    padding: int = 1  # Also allows tuple (padding_h, padding_w)
    R: int = 3
    S: int = 3
    has_bias: bool = False

    group_width = 8
    stride = 1

    @staticmethod
    def from_tensors(input, weight, bias, padding=1):
        """Return a Conv2dGw8Params instance from input, weight, and bias tensors.

        The tensors are arguments from a torch.nn.functional.conv2d call.

        Args:
            input (torch.Tensor): The input tensor.
            weight (torch.Tensor): The weight tensor.
            bias (torch.Tensor or None): The bias tensor or None.
            padding (int or tuple): Padding for height and width.
        """
        assert input.dtype == torch.float16
        assert weight.dtype == torch.float16
        assert bias is None or (
            len(bias.shape) == 1 and bias.shape[0] == weight.shape[0]
        )
        N, C, H, W = input.shape
        K, group_width, R, S = weight.shape
        assert (
            group_width == Conv2dGw8Params.group_width
        ), "Only group width of 8 is supported."
        assert K == C, "Number of output channels must match number of input channels."
        has_bias = bias is not None
        params = Conv2dGw8Params(
            N=N,
            C=C,
            H=H,
            W=W,
            R=R,
            S=S,
            padding=padding,
            has_bias=has_bias,
        )
        params.validate()
        return params

    @staticmethod
    def from_torch_module(module, input: torch.Tensor):
        N, C, H, W = input.shape
        K, group_width, R, S = module.weight.shape
        has_bias = module.bias is not None
        padding = module.padding
        return Conv2dGw8Params(
            N=N,
            C=C,
            H=H,
            W=W,
            R=R,
            S=S,
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
        assert self.N > 0
        assert self.C > 0
        assert self.H > 0
        assert self.W > 0
        assert self.padding_h >= 0
        assert self.padding_w >= 0
        assert (
            self.C % self.group_width == 0
        ), "Number of channels must be divisible by group width."
        assert self.R in range(1, 6), "Kernel height must be between 1 and 5."
        assert self.S in range(1, 6), "Kernel width must be between 1 and 5."

    @property
    def groups(self):
        """Return the number of groups."""
        return self.C // self.group_width

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
        return self.R - 1 - self.padding_h

    @property
    def transpose_padding_w(self):
        """Return the width padding of the transposed convolution."""
        return self.S - 1 - self.padding_w

    @property
    def K(self):
        """Return the number of output channels.

        Equal to the number of input channels.
        """
        return self.C

    @property
    def P(self):
        """Return the height of the output."""
        return self.H + 2 * self.padding_h - self.R + 1

    @property
    def Q(self):
        """Return the width of the output."""
        return self.W + 2 * self.padding_w - self.S + 1

    @property
    def kernel_size(self):
        """Return the kernel size as a tuple (R, S)."""
        return (self.R, self.S)

    @property
    def input_shape(self):
        """Return the shape of the input tensor as a tuple (N, C, H, W)."""
        return (self.N, self.C, self.H, self.W)

    @property
    def output_shape(self):
        """Return the shape of the output tensor as a tuple (N, C, P, Q)."""
        return (self.N, self.K, self.P, self.Q)

    @property
    def weight_shape(self):
        """Return the shape of the weight tensor as a tuple (K, group_width, R, S)."""
        return (self.K, self.group_width, self.R, self.S)

    @property
    def bias_shape(self):
        """Return the shape of the bias tensor as a tuple (K,)."""
        return (self.K,)
