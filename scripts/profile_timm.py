import argparse
import sys
from datetime import datetime

import torch
import timm
from timm.utils import ParseKwargs

import spio
import spio.log
import spio.transform


def get_trace_file_name(args, datestamp, ext=".json"):
    memory_format_name = "channels_last" if args.channels_last else "channels_first"
    filename = f"trace_{args.model}_{memory_format_name}_bs{args.batch_size}"
    if args.torchcompile:
        filename += f"_compiled"
    if args.spio:
        filename += "_spio"
    filename += f"_{datestamp}"
    filename += ext
    return filename


NUM_INPUTS = 4
WAIT_ITERS = 1

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="convnext_base")
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--channels-last", action="store_true")
parser.add_argument("--torchcompile", action="store_true")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--warmup-iters", type=int, default=10)
parser.add_argument("--benchmark-iters", type=int, default=10)
parser.add_argument("--summary", action="store_true")
parser.add_argument("--spio", action="store_true")
parser.add_argument("--model-kwargs", nargs="*", default={}, action=ParseKwargs)
parser.add_argument("--no-profile", action="store_true")
parser.add_argument("--enable-logger", action="store_true")
parser.add_argument("--scan-modules", action="store_true")

args = parser.parse_args()

torch.backends.cudnn.benchmark = True

spio.log.benchmark_logger.configure(
    full_format=True,
    best=False,
    header_once=True,
    enable=args.enable_logger,
)

memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
total_iters = WAIT_ITERS + args.warmup_iters + args.benchmark_iters

model = timm.create_model(
    args.model, pretrained=args.pretrained, **args.model_kwargs
).cuda()

training_input_size = model.default_cfg["input_size"]

inputs = [
    torch.randn((args.batch_size, *training_input_size), device="cuda").to(
        memory_format=memory_format, dtype=torch.float16
    )
    for _ in range(NUM_INPUTS)
]

model.to(device="cuda", memory_format=memory_format)

if args.summary:
    try:
        import torchinfo
    except ImportError as e:
        raise ImportError(
            "torchinfo is required for --summary. Please install it using 'pip install torchinfo'."
        ) from e

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

if args.scan_modules:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        params_lst = spio.transform.scan_modules(model, inputs[0])
        for params in params_lst:
            print(params)
        sys.exit(0)

if args.torchcompile:
    model = torch.compile(model)

if args.spio:
    model = spio.transform.transform(model)

if args.no_profile:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(inputs[0]).sum()
    out.backward()
else:
    schedule = torch.profiler.schedule(
        wait=WAIT_ITERS, warmup=args.warmup_iters, active=args.benchmark_iters, repeat=1
    )

    with torch.profiler.profile(schedule=schedule) as prof:
        for i in range(total_iters):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(inputs[i % NUM_INPUTS]).sum()
            out.backward()
            prof.step()

    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    trace_file_name = get_trace_file_name(args, datestamp)
    prof.export_chrome_trace(trace_file_name)
    print("Wrote trace to ", trace_file_name)

    data_file_name = get_trace_file_name(args, datestamp, ext=".dat")
    with open(data_file_name, "w") as f:
        f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    print("Wrote data to ", data_file_name)
