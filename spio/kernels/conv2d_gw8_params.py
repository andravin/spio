from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Conv2dGw8Params:
    N: int
    C: int
    H: int
    W: int
    padding: int = 1  # Also allows tuple (padding_h, padding_w)
    R: int = 3
    S: int = 3
    group_width: int = 8
    has_bias: bool = False

    @staticmethod
    def from_tensors(input, weight, bias, padding=1):
        assert input.dtype == torch.float16
        assert weight.dtype == torch.float16
        assert bias is None or (len(bias.shape) == 1 and bias.shape[0] == weight.shape[0])
        N, C, H, W = input.shape
        K, group_width, R, S = weight.shape
        has_bias = bias is not None
        params = Conv2dGw8Params(
            N=N,
            C=C,
            H=H,
            W=W,
            R=R,
            S=S,
            padding=padding,
            group_width=group_width,
            has_bias=has_bias,
        )
        params.validate()
        return params

    def is_valid(self) -> bool:
        try:
            self.validate()
            return True
        except AssertionError:
            return False

    def validate(self):
        assert self.group_width == 8
        assert self.N > 0
        assert self.C > 0
        assert self.H > 0
        assert self.W > 0
        assert self.padding_h >= 0
        assert self.padding_w >= 0
        assert self.C % self.group_width == 0
        assert self.R in range(6)
        assert self.S in range(6)

    @property
    def groups(self):
        return self.C // self.group_width

    @groups.setter
    def groups(self, value):
        if self.C % value != 0:
            raise ValueError(f"Number of groups must divide the number of channels")
        group_width = self.C // value
        if group_width != self.group_width:
            raise ValueError(f"Group width must be {self.group_width}")

    @property
    def padding_h(self):
        return self.padding[0] if isinstance(self.padding, tuple) else self.padding

    @property
    def padding_w(self):
        return self.padding[1] if isinstance(self.padding, tuple) else self.padding

    @property
    def transpose_padding_h(self):
        return self.R - 1 - self.padding_h

    @property
    def transpose_padding_w(self):
        return self.S - 1 - self.padding_w

    @property
    def P(self):
        return self.H + 2 * self.padding_h - self.R + 1

    @property
    def Q(self):
        return self.W + 2 * self.padding_w - self.S + 1

    @property
    def kernel_size(self):
        return (self.R, self.S)

    @property
    def input_shape(self):
        return (self.N, self.C, self.H, self.W)

    @property
    def output_shape(self):
        return (self.N, self.C, self.P, self.Q)

    @property
    def weight_shape(self):
        return (self.C, self.group_width, self.R, self.S)

    @property
    def bias_shape(self):
        return (self.C,)

    @property
    def wgrad_shape(self):
        return self.weight_shape

    @property
    def deltas_shape(self):
        return self.output_shape

    @property
    def igrad_shape(self):
        return self.input_shape

    # TODO move this method to the Reflection class.
    @property
    def kwargs(self):
        return dict(stride=1, padding=self.padding, groups=self.groups)
