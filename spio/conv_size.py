"""Create a conv2d layer and convert to channels-last memory_format to inspect the weights order.

This is a diagnostic for inspecting the PyTorch's weights format.
"""

import torch
from torch import nn

BATCH_SIZE = 4
HEIGHT = 256
WIDTH = 256
GROUP_WIDTH = 4
CHS_IN = 32  # C
CHS_OUT = 64 # K

if GROUP_WIDTH is None:
    groups = 1
else:
    groups = CHS_IN // GROUP_WIDTH

# conv2d.weight constructed in [K C R S] order (channels-first).
layer = nn.Conv2d(CHS_IN, CHS_OUT, kernel_size=3, groups=groups, padding=1)
layer = layer.cuda()

# conv2d.weight converted to [K R S C] order (channels-last).
layer = layer.to(memory_format=torch.channels_last)

inputs = [
    torch.randn((BATCH_SIZE, CHS_IN, HEIGHT, WIDTH), device="cuda").to(
        memory_format=torch.channels_last
    )
    for _ in range(10)
]

with torch.autocast(device_type="cuda", dtype=torch.float16):
    for i in range(1, 4):
        out = layer(inputs[i])
