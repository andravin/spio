import torch
from torch import nn
from torchvision.models import resnet18

BATCH_SIZE = 128
HEIGHT = 256
WIDTH = 256
GROUP_WIDTH = 8

class GroupedConvFiesta(nn.Module):
    def __init__(self, num_channels=64, group_width=8, depth=16):
        super().__init__()
        groups = num_channels // group_width
        self.stem = nn.Conv2d(3, num_channels, kernel_size=4, stride=4, padding=0)
        convs = [
            nn.Conv2d(
                num_channels, num_channels, kernel_size=3, groups=groups, padding=1
            )
            for d in range(depth)
        ]
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        x = self.stem(x)
        x = self.convs(x)
        return x


# model = resnet18().cuda()
model = GroupedConvFiesta(group_width=GROUP_WIDTH).cuda()
model = model.to(memory_format=torch.channels_last)
inputs = [
    torch.randn((BATCH_SIZE, 3, HEIGHT, WIDTH), device="cuda").to(
        memory_format=torch.channels_last
    )
    for _ in range(10)
]

model_c = torch.compile(model)


def fwd_bwd(inp):
    out = model_c(inp)
    out.sum().backward()


# warm up
with torch.autocast(device_type="cuda", dtype=torch.float16):
    fwd_bwd(inputs[0])

    with torch.profiler.profile() as prof:
        for i in range(1, 4):
            fwd_bwd(inputs[i])
            prof.step()

    prof.export_chrome_trace("trace.json")
