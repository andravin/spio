import argparse
import sys
from pathlib import Path
import datetime
import random
from math import prod
from io import StringIO, BytesIO
from typing import Dict, List

import torch
from torch import nn
import pandas as pd
from tqdm import tqdm

try:
    import timm

    has_timm = True
except ImportError:
    has_timm = False

try:
    import torchinfo

    has_torchinfo = True
except ImportError:
    has_torchinfo = False

from spio.kernels import KernelParams
from spio.transform import transform as spio_transform
from spio.util import get_formatted_device_name, ParseKwargs, Timer
from spio.kernels import (
    KernelParamsLogger,
    KernelParams,
    KernelKey,
    Kernel,
)
from spio.compiler import compile_kernels
from spio.src_tests import preprocess_data_string

GROUP_WIDTH = 8
CHANNELS_LAST = True
WARMUP_ITERS = 5
BENCHMARK_ITERS = 5
TOTAL_ITERS = WARMUP_ITERS + BENCHMARK_ITERS
CHANNELS = 64

RESULTS_TABLE_MAX_COL_WIDTH = 400
CONVOLUTIONAL_BLOCKS = ["MBConv", "ConvFirst"]


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        group_width=8,
        expansion_ratio=4,
        act_cls=nn.SiLU,
    ):
        padding = kernel_size // 2
        mid_channels = in_channels * expansion_ratio
        groups = mid_channels // group_width
        super().__init__()
        exp = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        exp_bn = nn.BatchNorm2d(mid_channels)
        exp_act = act_cls()

        conv = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=False,
        )
        conv_bn = nn.BatchNorm2d(mid_channels)
        conv_act = act_cls()

        prj = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        prj_bn = nn.BatchNorm2d(out_channels)

        self.layers = nn.Sequential(
            exp, exp_bn, exp_act, conv, conv_bn, conv_act, prj, prj_bn
        )
        self.has_residual = (in_channels == out_channels) and (stride == 1)

    def forward(self, x):
        input = x
        x = self.layers(x)
        if self.has_residual:
            x = x + input
        return x


class ConvFirst(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        group_width=8,
        expansion_ratio=4,
        act_cls=nn.SiLU,
    ):
        padding = kernel_size // 2
        mid_channels = in_channels * expansion_ratio
        groups = in_channels // group_width
        super().__init__()

        conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=False,
        )
        conv_bn = nn.BatchNorm2d(in_channels)

        exp = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        exp_bn = nn.BatchNorm2d(mid_channels)
        exp_act = act_cls()

        prj = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        prj_bn = nn.BatchNorm2d(out_channels)

        self.layers = nn.Sequential(conv, conv_bn, exp, exp_bn, exp_act, prj, prj_bn)
        self.has_residual = (in_channels == out_channels) and (stride == 1)

    def forward(self, x):
        input = x
        x = self.layers(x)
        if self.has_residual:
            x = x + input
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=4, act_cls=nn.SiLU):
        mid_channels = in_channels * expansion_ratio
        super().__init__()
        exp = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        exp_bn = nn.BatchNorm2d(mid_channels)
        exp_act = act_cls()

        prj = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        prj_bn = nn.BatchNorm2d(out_channels)

        self.layers = nn.Sequential(exp, exp_bn, exp_act, prj, prj_bn)
        self.has_residual = in_channels == out_channels

    def forward(self, x):
        input = x
        x = self.layers(x)
        if self.has_residual:
            x = x + input
        return x


class GroupedConvFiesta(nn.Module):
    def __init__(
        self,
        num_channels=64,
        kernel_size=3,
        group_width=8,
        expansion_ratio=4,
        depth=16,
        block_cls=MBConv,
        act_cls=nn.SiLU,
        extra_depth=1,
    ):
        super().__init__()

        blocks = [
            block_cls(
                num_channels,
                num_channels,
                kernel_size=kernel_size,
                group_width=group_width,
                expansion_ratio=expansion_ratio,
                act_cls=act_cls,
            )
            for _ in range(depth)
        ]

        extra = [
            MLP(
                num_channels,
                num_channels,
                expansion_ratio=expansion_ratio,
                act_cls=act_cls,
            )
            for _ in range(extra_depth)
        ]

        self.blocks = nn.Sequential(*blocks)
        self.extra = nn.Sequential(*extra)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.extra(x)
        x = self.pool(x)
        return x


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--group-width", type=int, default=GROUP_WIDTH)
    parser.add_argument("--channels", type=int, default=CHANNELS)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="The number of convolutional blocks in the network.",
    )
    parser.add_argument(
        "--extra",
        type=int,
        default=1,
        help="The number of extra MLP blocks to append to the network.",
    )
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--spio", action="store_true")
    parser.add_argument("--corpus-size", type=int, default=4)
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--group-by-input-shape", action="store_true")
    parser.add_argument("--expansion-ratio", type=int, default=4)
    parser.add_argument(
        "--block",
        type=str,
        default="MBConv",
        choices=CONVOLUTIONAL_BLOCKS,
        help="The type of convolutional block to use.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-start", type=int, default=None)
    parser.add_argument("--batch-end", type=int, default=None)
    parser.add_argument("--batch-step", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--benchmark-configs", action="store_true")
    parser.add_argument("--timm-model", type=str, default=None)
    parser.add_argument(
        "--timm-model-kwargs", nargs="*", default={}, action=ParseKwargs
    )
    parser.add_argument(
        "--max-random-samples",
        type=int,
        default=200,
        help="The maximum number of random samples to take from the kernel configurations.",
    )
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    if args.output_dir is None:
        args.output_dir = "."
    args.output_dir = Path(args.output_dir) / get_dir_name(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_start is not None and args.batch_end is not None:
        for batch_size in range(args.batch_start, args.batch_end, args.batch_step):
            args.batch_size = batch_size
            run_benchmark(args, batch_size)
    else:
        run_benchmark(args)


def run_benchmark(args, batch_size: int = None):
    if batch_size is not None:
        args.batch_size = batch_size

    if args.block == "MBConv":
        block_cls = MBConv
    elif args.block == "ConvFirst":
        block_cls = ConvFirst
    else:
        raise ValueError(f"Unknown block type: {args.block}")

    device = torch.device(f"cuda:{args.device}")
    if args.timm_model is not None:
        if not has_timm:
            print(
                "The --timm-model option requires timm to be installed.",
                file=sys.stderr,
            )
            sys.exit(1)
        model = timm.create_model(
            args.timm_model, pretrained=False, **args.timm_model_kwargs
        ).cuda(args.device)
        input_size = model.default_cfg["input_size"]
    else:
        model = GroupedConvFiesta(
            group_width=args.group_width,
            kernel_size=args.kernel_size,
            depth=args.depth,
            extra_depth=args.extra,
            num_channels=args.channels,
            expansion_ratio=args.expansion_ratio,
            block_cls=block_cls,
        ).cuda(args.device)
        input_size = (args.channels, args.resolution, args.resolution)

    model.to(memory_format=torch.channels_last)
    input_shape = (args.batch_size, *input_size)
    inputs = [
        torch.randn(input_shape, device=device).to(memory_format=torch.channels_last)
        for _ in range(args.corpus_size)
    ]

    if args.summary:
        if not has_torchinfo:
            print(
                "The --summary summary option requires torchinfo to be installed.",
                file=sys.stderr,
            )
            sys.exit(1)

        torchinfo.summary(
            model,
            input_size=input_shape,
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

    if args.spio:
        model = spio_transform(model)

    if args.compile:
        model = torch.compile(model)

    if args.benchmark_configs:
        output_path = args.output_dir / get_benchmark_model_output_file_name(args)
        print("Will save benchmark results to", output_path)
        benchmark_configs(model, inputs, args, output_path)
        sys.exit(0)

    # Run the benchmark with PyTorch.
    schedule = torch.profiler.schedule(
        wait=1, warmup=WARMUP_ITERS - 1, active=BENCHMARK_ITERS, repeat=1
    )
    activities = [
        torch.profiler.ProfilerActivity.CUDA,
    ]
    if args.with_stack or args.group_by_input_shape:
        activities.append(torch.profiler.ProfilerActivity.CPU)
        experimental_config = torch._C._profiler._ExperimentalConfig(verbose=True)
    else:
        experimental_config = None

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        with_stack=args.with_stack,
        experimental_config=experimental_config,
        record_shapes=args.group_by_input_shape,
    ) as prof:
        for i in range(TOTAL_ITERS):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(inputs[i % args.corpus_size])
            out.sum().backward()
            if i == WARMUP_ITERS - 1:
                torch.cuda.synchronize()
            prof.step()

    if args.batch_start is not None and args.batch_end is not None:
        file_name_details = get_batch_size_file_name_details(args)
    else:
        file_name_details = get_file_name_details(args)
    file_name = args.output_dir / f"bench_{file_name_details}"

    if args.save_trace:
        json_file_name = str(file_name.with_suffix(".json"))
        prof.export_chrome_trace(json_file_name)
        print(f"Trace saved to {json_file_name}")

    data_file_name = str(file_name.with_suffix(".dat"))
    group_by_stack_n = 8 if args.with_stack else 0
    with open(data_file_name, "w") as f:
        f.write(
            prof.key_averages(
                group_by_input_shape=args.group_by_input_shape,
                group_by_stack_n=group_by_stack_n,
            ).table(
                sort_by="self_cuda_time_total",
                row_limit=-1,
                max_src_column_width=RESULTS_TABLE_MAX_COL_WIDTH,
                max_name_column_width=RESULTS_TABLE_MAX_COL_WIDTH,
            )
        )
        print(f"Averages saved to {data_file_name}")


def benchmark_configs(model, inputs, args, data_path: Path):
    # Run the model to trace the kernel parameters.
    with KernelParamsLogger() as logger:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(inputs[0])
        out.sum().backward()
        logged_kernel_params = logger.get_logged_params()
        unique_kernel_params = set(logged_kernel_params)

    # Clear the kernel caches from the configs that were used during logging.
    unique_kernel_caches = set(
        kernel_params.kernel_cache for kernel_params in unique_kernel_params
    )
    for kernel_cache in unique_kernel_caches:
        kernel_cache.clear_cache()

    # For each unique KernelParams, compile all kernel configurations.
    # Store the kernels in a table.
    kernel_table: Dict[KernelParams, List[Kernel]] = dict()
    for kernel_params in unique_kernel_params:
        kernel_factory = kernel_params.kernel_factory
        params = kernel_params.params
        device = kernel_params.device
        arch = torch.cuda.get_device_capability(device=device)
        configs = kernel_factory.configs(kernel_params.params)
        kernel_kwargs = dict(kernel_params.kernel_kwargs)
        kernel_table[kernel_params] = [
            kernel_factory.make_kernel(params, config, **kernel_kwargs)
            for config in configs
        ]

    # Benchmark args.max_random_samples kernel configurations per unique layer params.
    # Some kernels may have more configurations than others.
    # Therefore some kernels will be benchmarked multiple times.
    max_configs = max(len(kernel_lst) for kernel_lst in kernel_table.values())
    num_choices = min(max_configs, args.max_random_samples)

    all_kernel_choices = dict()
    for kernel_params, kernel_lst in kernel_table.items():
        all_kernel_choices[kernel_params] = random_sample_with_replacement(
            kernel_lst, num_choices
        )

    kernels_to_compile = []
    for kernel_params, kernel_lst in all_kernel_choices.items():
        num_unique_configs = len(kernel_table[kernel_params])
        kernels_to_compile.extend(kernel_lst[:num_unique_configs])

    with Timer(f"Compiling {len(kernels_to_compile)} kernels"):
        compile_kernels(kernels_to_compile, arch=arch)

    with data_path.open("w") as f:

        f.write(f"Kernel;Params;Config;KernelKwargs;CUDA_time_avg_ms\n")

        for idx in tqdm(range(num_choices), desc=f"Profiling kernel configurations"):
            # Select a random combination of kernel configurations without replacement.
            kernel_choices: Dict[KernelParams, Kernel] = dict()
            for kernel_params, kernel_lst in all_kernel_choices.items():
                kernel_choices[kernel_params] = kernel_lst[idx]

            # Create cache overlays for each KernelCache.
            kernel_cache_overlays = dict()
            for kernel_params in kernel_choices.keys():
                kernel_cache_overlays[kernel_params.kernel_cache] = dict()
            for kernel_params, kernel in kernel_choices.items():
                kernel_cache = kernel_params.kernel_cache
                key: KernelKey = kernel_params.key
                kernel_cache_overlays[kernel_cache][key] = kernel
            for kernel_cache, overlay in kernel_cache_overlays.items():
                kernel_cache.update_overlay(overlay)

            # Load the kernels
            for kernel in kernel_choices.values():
                if kernel.cubin is None:
                    # Sanity check to ensure that the kernel was compiled and not unloaded.
                    # This should never fail.
                    raise ValueError("Kernel cubin is None")
                # Load the kernel now and allow it to be reloaded later.
                kernel.load(device.index, clear_cubin=False)

            # Run the benchmark.
            schedule = torch.profiler.schedule(
                wait=1, warmup=WARMUP_ITERS - 1, active=BENCHMARK_ITERS, repeat=1
            )
            activities = [
                torch.profiler.ProfilerActivity.CUDA,
            ]
            with torch.profiler.profile(
                activities=activities, schedule=schedule
            ) as prof:
                for i in range(TOTAL_ITERS):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = model(inputs[i % args.corpus_size])
                    out.sum().backward()
                    if i == WARMUP_ITERS - 1:
                        torch.cuda.synchronize()
                    prof.step()
            torch.cuda.synchronize()

            # Clear the overlays
            for kernel_cache in kernel_cache_overlays.keys():
                kernel_cache.clear_overlay()

            # Unload the kernels
            for kernel in kernel_choices.values():
                kernel.unload()

            # Get the benchmark results.
            table = prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=-1,
                max_src_column_width=RESULTS_TABLE_MAX_COL_WIDTH,
                max_name_column_width=RESULTS_TABLE_MAX_COL_WIDTH,
            )
            table_ssv_lines = preprocess_data_string(table)
            table_ssv_str = "\n".join(table_ssv_lines)

            data_io = StringIO(table_ssv_str)

            df = pd.read_csv(data_io, sep=";")

            # Extract the kernel timings from the benchmark results.
            for kernel_params, kernel in kernel_choices.items():
                kernel_name = kernel.kernel_name
                avg_time_ms = df.loc[df["Name"] == kernel_name, "CUDA_time_av"].values[
                    0
                ]
                params = kernel.params
                config = kernel.config
                kernel_kwargs = str(kernel_params.kernel_kwargs)
                out_str = (
                    f"{kernel_name};{params};{config};{kernel_kwargs};{avg_time_ms}"
                )
                f.write(out_str + "\n")


def get_dir_name(args):
    device_name = get_formatted_device_name(f"cuda:{args.device}")
    if args.timm_model is not None:
        model_name = args.timm_model
        kwargs_str = "__".join(
            [f"{key}__{value}" for key, value in args.timm_model_kwargs.items()]
        )
        return f"modelbench___{device_name}___{model_name}___{kwargs_str}"
    else:
        file_name = get_file_name_details(args, no_bs=True)
        date_stamp = get_date_stamp()
        return f"bench__{device_name}__{file_name}__{date_stamp}"


def get_file_name_details(args, no_bs=False):
    backend = "spio" if args.spio else "torch"
    filename = f"{backend}_{args.block.lower()}_c{args.channels}_ks{args.kernel_size}_er{args.expansion_ratio}_gw{args.group_width}_hw{args.resolution}_d{args.depth}_extra{args.extra}_iters{BENCHMARK_ITERS}"
    if not no_bs:
        filename += f"_bs{args.batch_size}"
    if args.compile:
        filename += "_compiled"
    if args.with_stack:
        filename += "_stack"
    return filename


def get_benchmark_model_output_file_name(args):
    """Automatically generate a file name for the model benchmark results."""
    if args.timm_model is not None:
        date_stamp = get_date_stamp()
        return f"model_bench_{args.batch_size}__{date_stamp}.ssv"
    else:
        raise NotImplementedError(
            "TODO: Implement output file name for block profiling results"
        )


def get_batch_size_file_name_details(args):
    return f"bs{args.batch_size}"


def get_date_stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def random_sample_with_replacement(alist, k):
    """Randomly sample k items from a list with replacement.

    If k is greater than the length of the list, the list will be sampled multiple times.

    Args:
        alist (List): The list to sample from.
        k (int): The number of samples to take.

    Returns:
        List: A list of k randomly sampled items from alist."""
    n = len(alist)
    result = []

    # Calculate the number of full cycles
    full_cycles = k // n
    remaining_items = k % n

    # Add full cycles
    for _ in range(full_cycles):
        shuffled_list = alist[:]
        random.shuffle(shuffled_list)
        result.extend(shuffled_list)

    # Add remaining items
    if remaining_items > 0:
        shuffled_list = alist[:]
        random.shuffle(shuffled_list)
        result.extend(shuffled_list[:remaining_items])

    return result


if __name__ == "__main__":
    main()
