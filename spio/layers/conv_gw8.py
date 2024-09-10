import torch
from torch import nn

from ..functional import conv2d_gw8
from ..kernels import Conv2dGw8Params


class Conv2dGw8(nn.Conv2d):
    Params = Conv2dGw8Params

    @staticmethod
    def match(conv2d: nn.Conv2d):
        if not isinstance(conv2d, nn.Conv2d):
            return False
        group_width = conv2d.in_channels // conv2d.groups
        R, S = conv2d.kernel_size
        return (
            group_width == 8
            and R >= 1
            and R <= 5
            and S >= 1
            and S <= 5
            and conv2d.in_channels == conv2d.out_channels
            and conv2d.stride == (1, 1)
            and conv2d.dilation == (1, 1)
        )

    @staticmethod
    def from_torch_module(conv2d: nn.Conv2d):
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
            dtype=torch.float16,
        ).to(memory_format=torch.channels_last)
        module.weight.data.copy_(conv2d.weight)
        if conv2d.bias is not None:
            module.bias.data.copy_(conv2d.bias)
        return module

    def forward(self, x):
        padding = self.padding
        if isinstance(padding, int):
            padding_y = padding
            padding_x = padding
        else:
            padding_y, padding_x = padding

        return conv2d_gw8(
            x,
            self.weight,
            self.bias,
            stride=1,
            padding_y=padding_y,
            padding_x=padding_x,
            groups=self.groups,
        )
