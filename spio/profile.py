import argparse

import torch
from torch import nn
import cupy

BATCH_SIZE = 128
HEIGHT = 256
WIDTH = 256
GROUP_WIDTH = 4
CHANNELS_LAST = True
WARMUP_ITERS = 10
BENCHMARK_ITERS = 1
TOTAL_ITERS = WARMUP_ITERS + BENCHMARK_ITERS
DEPTH = 1
CHANNELS = 64
CUDA_PROFILE = False
USE_BIAS = False


parser = argparse.ArgumentParser()
parser.add_argument("--group-width", type=int, default=GROUP_WIDTH)
parser.add_argument("--channels", type=int, default=CHANNELS)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--channels-first", action="store_true")
args = parser.parse_args()

memory_format = torch.contiguous_format if args.channels_first else torch.channels_last
memory_format_name = (
    "contiguous" if memory_format == torch.contiguous_format else "channels_last"
)


class GroupedConvFiesta(nn.Module):
    def __init__(self, num_channels=64, group_width=8, depth=16, bias=True):
        super().__init__()
        groups = num_channels // group_width
        self.stem = nn.Conv2d(3, num_channels, kernel_size=4, stride=4, padding=0)
        convs = [
            nn.Sequential(
                nn.Conv2d(
                    num_channels, num_channels, kernel_size=3, groups=groups, padding=1
                ),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(),
            )
            for d in range(depth)
        ]
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        x = self.stem(x)
        x = self.convs(x)
        return x


model = GroupedConvFiesta(
    group_width=args.group_width, depth=DEPTH, num_channels=args.channels, bias=USE_BIAS
).cuda()

model = model.to(memory_format=memory_format)
inputs = [
    torch.randn((BATCH_SIZE, 3, HEIGHT, WIDTH), device="cuda").to(
        memory_format=memory_format
    )
    for _ in range(TOTAL_ITERS)
]

if args.compile:
    model = torch.compile(model)


def fwd_bwd(inp):
    out = model(inp)
    out.sum().backward()


with torch.autocast(device_type="cuda", dtype=torch.float16):
    # warm up
    for i in range(WARMUP_ITERS):
        fwd_bwd(inputs[i])

    if CUDA_PROFILE:
        # Run the benchmark with a CUDA profiler section.
        cupy.cuda.runtime.profilerStart()
        for i in range(WARMUP_ITERS, TOTAL_ITERS):
            fwd_bwd(inputs[i])
        cupy.cuda.runtime.profilerStop()
    else:
        # Run the benchmark with PyTorch.
        with torch.profiler.profile() as prof:
            for i in range(WARMUP_ITERS, TOTAL_ITERS):
                fwd_bwd(inputs[i])
                prof.step()
        compiled = "compiled" if args.compile else "uncompiled"
        prof.export_chrome_trace(
            f"trace_gw{args.group_width}_{memory_format_name}_{compiled}.json"
        )
