import argparse
import sys

import torch
import timm

import spio


def trace_file_name(args):
    memory_format_name = "channels_last" if args.channels_last else "channels_first"
    compiled = "compiled" if args.compile else "not-compiled"
    filename = f"trace_{args.model}_{memory_format_name}_bs{args.batch_size}"
    if args.compile:
        filename += f"_compiled"
    if args.spio:
        filename += "_spio"
    filename += ".json"
    return filename


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="convnext_base")
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--channels-last", action="store_true")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--warmup-iters", type=int, default=10)
parser.add_argument("--benchmark-iters", type=int, default=1)
parser.add_argument("--summary", action="store_true")
parser.add_argument("--spio", action="store_true")

args = parser.parse_args()

torch.backends.cudnn.benchmark = True

memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
total_iters = args.warmup_iters + args.benchmark_iters

model = timm.create_model(args.model, pretrained=args.pretrained).cuda()

training_input_size = model.default_cfg["input_size"]

inputs = [
    torch.randn((args.batch_size, *training_input_size), device="cuda").to(
        memory_format=memory_format, dtype=torch.float16
    )
    for _ in range(total_iters)
]

model = model.to(memory_format=memory_format)

if args.summary:
    import torchinfo
    torchinfo.summary(
        model,
        input_size=(args.batch_size, *training_input_size),
        depth=99,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "params_percent",
            "kernel_size",
            "groups",
            "mult_adds",
            "trainable",
        ],
    )
    sys.exit(0)

if args.compile:
    model = torch.compile(model)

if args.spio:
    model = spio.replace_ops(model)

for i in range(args.warmup_iters):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(inputs[i]).sum()
    out.backward()

with torch.profiler.profile() as prof:
    for i in range(args.warmup_iters, total_iters):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(inputs[i]).sum()
        out.backward()
        prof.step()

prof.export_chrome_trace(trace_file_name(args))
