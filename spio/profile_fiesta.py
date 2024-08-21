import argparse
import sys

import torch
from torch import nn
import cupy

GROUP_WIDTH = 4
CHANNELS_LAST = True
WARMUP_ITERS = 10
BENCHMARK_ITERS = 1
TOTAL_ITERS = WARMUP_ITERS + BENCHMARK_ITERS
CHANNELS = 64
CUDA_PROFILE = False


parser = argparse.ArgumentParser()
parser.add_argument("--group-width", type=int, default=GROUP_WIDTH)
parser.add_argument("--channels", type=int, default=CHANNELS)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--channels-first", action="store_true")
parser.add_argument("--use-bias", action="store_true")
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--depth", type=int, default=1)
parser.add_argument("--summary", action="store_true")
parser.add_argument("--batch-size", type=int, default=128)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True

memory_format = torch.contiguous_format if args.channels_first else torch.channels_last
memory_format_name = (
    "contiguous" if memory_format == torch.contiguous_format else "channels_last"
)


class GroupedConvFiesta(nn.Module):
    def __init__(self, num_channels=64, group_width=8, depth=16, bias=False, expansion=4):
        super().__init__()
        groups = num_channels // group_width
        mid_channels = num_channels * expansion
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=4, stride=4, padding=0, bias=bias),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        convs = [
            nn.Sequential(
                nn.Conv2d(
                    num_channels, num_channels, kernel_size=3, groups=groups, padding=1, bias=bias
                ),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(),
                nn.Conv2d(num_channels, mid_channels, kernel_size=1, bias=bias),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, num_channels, kernel_size=1, bias=bias),
                nn.BatchNorm2d(num_channels),
            )
            for d in range(depth)
        ]
        self.convs = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.convs(x)
        x = self.pool(x)
        return x


model = GroupedConvFiesta(
    group_width=args.group_width, depth=args.depth, num_channels=args.channels, bias=args.use_bias
).cuda()

model = model.to(memory_format=memory_format)
input_size = (3, args.resolution, args.resolution)
inputs = [
    torch.randn((args.batch_size, *input_size), device="cuda", dtype=torch.float16).to(
        memory_format=memory_format
    )
    for _ in range(TOTAL_ITERS)
]

if args.summary:
    import torchinfo
    torchinfo.summary(
        model,
        input_size=(args.batch_size, *input_size),
        depth=99,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "params_percent",
            "kernel_size",
            "mult_adds",
            "trainable",
        ],
    )
    sys.exit(0)


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
            f"fiesta2_gw{args.group_width}_d{args.depth}_c{args.channels}_hw{args.resolution}-{memory_format_name}_{compiled}.json"
        )
