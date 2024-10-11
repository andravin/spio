import argparse
import sys
from typing import List
import time
from dataclasses import replace
import random

import torch
import ast

import spio.functional
import spio.kernels
import spio.functional
import spio.benchmark
import spio.reflection
from spio.log import (
    BenchmarkResultCompactFormat,
    BenchmarkResultFullFormat,
)
from spio.reflection import get_function_reflection, get_kernel_reflection
from spio.kernels import Conv2dGw8Params
from spio.util import (
    get_formatted_arch,
    get_formatted_device_name,
    ParseKwargs,
)

from spio.util.load_parameter_set import load_parameter_set

PARAMS_CLASSES = {"Conv2dGw8Params": Conv2dGw8Params}


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--kernel", type=str, default=None)
    g.add_argument("--function", type=str, default=None)
    parser.add_argument("--params", nargs="*", action=ParseKwargs)
    parser.add_argument(
        "--params-set", action="store_true", help="Use default parameter set"
    )
    parser.add_argument("--config", nargs="*", default={}, action=ParseKwargs)
    parser.add_argument("--kernel-kwargs", nargs="*", default={}, action=ParseKwargs)
    parser.add_argument("--auto-tune", action="store_true")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--benchmark-torch", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--full-format", action="store_true")
    parser.add_argument("--grad-inputs", nargs="*", default=None)
    parser.add_argument("--write-to-file", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--randomize-batch-size", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ssv",
        action="store_true",
        help="Output in semicolon-separated variable format",
    )
    args = parser.parse_args()

    # Enabling cuddn benchmarks is essential for good PyTorch performance.
    torch.backends.cudnn.benchmark = True

    if args.write_to_file:
        output_file = get_output_file_name(args)
        print("Writing output to", output_file)
        sys.stdout = open(output_file, "w")

    format_cls = (
        BenchmarkResultFullFormat if args.full_format else BenchmarkResultCompactFormat
    )
    fmt = format_cls(delim=";" if args.ssv else None)

    print(fmt.header())

    prefix_op = None

    if args.randomize_batch_size:
        random.seed(args.seed)

    if args.kernel is not None:
        kernel_class = getattr(spio.kernels, args.kernel)
        if args.params_set:
            params_lst = load_parameter_set(kernel_class.Params)
        else:
            params = kernel_class.Params(**args.params)
            params_lst = [params]
        reflection = get_kernel_reflection(kernel_class, **args.kernel_kwargs)

        for epoch in range(args.epochs):
            for params in params_lst:

                if args.randomize_batch_size:
                    params = replace(params, N=randomize_batch_size(params.N))

                if reflection.prefix_op_constructor is not None:
                    prefix_op = reflection.prefix_op_constructor(params)

                if len(args.config) == 0:
                    configs = None
                else:
                    configs = [kernel_class.Config(**args.config)]

                if reflection.stackable:
                    args_lst = reflection.make_stacked_args(
                        params, depth=args.depth, device=args.device
                    )
                else:
                    args_lst = [
                        reflection.make_args(params, device=args.device)
                        for _ in range(args.depth)
                    ]

                input_names = None

                kernel_args_lst = [reflection.arrange_args(args) for args in args_lst]
                best_result, best_kernel, results = spio.kernels.benchmark_kernel(
                    kernel_class,
                    params,
                    kernel_args_lst,
                    configs=configs,
                    warmup=args.warmup,
                    num_iters=args.num_iters,
                    device=args.device,
                    prefix_op=prefix_op,
                    **args.kernel_kwargs,
                )
                best_kernel.unload()
                torch.cuda.synchronize()
                print(fmt.results(results, sort=True, header=False))
                output_names = best_kernel.output_names
                is_backprop = best_kernel.is_backprop
                if is_backprop:
                    input_names = [
                        reflection.get_arg_name_from_gradient(grad_name)
                        for grad_name in output_names
                    ]
    elif args.function is not None:
        function = getattr(spio.functional, args.function)
        reflection = get_function_reflection(function)
        is_backprop = args.grad_inputs is not None
        input_names = args.grad_inputs
        params = reflection.params(**args.params)
        if reflection.prefix_op_constructor is not None:
            prefix_op = reflection.prefix_op_constructor(params)
        args_lst = [
            reflection.make_args(params, device=args.device, training=is_backprop)
            for _ in range(args.depth)
        ]
        profile_function(
            function=function,
            is_backprop=is_backprop,
            input_names=input_names,
            params=params,
            args=args,
            args_lst=args_lst,
            prefix_op=prefix_op,
        )

    if args.benchmark_torch:
        if reflection.reference is None:
            print(f"No torch function defined for {reflection}.")
            sys.exit(1)
        args_lst = [
            reflection.make_args(params, device=args.device, training=is_backprop)
            for _ in range(args.depth)
        ]
        profile_function(
            function=reflection.reference,
            is_backprop=is_backprop,
            input_names=input_names,
            params=params,
            args=args,
            args_lst=args_lst,
            prefix_op=prefix_op,
        )


def profile_function(
    function=None,
    is_backprop=False,
    input_names=None,
    params=None,
    args=None,
    args_lst=None,
    prefix_op=None,
):
    function_reflection = get_function_reflection(function)
    args_lst = function_reflection.arrange_stacked_args_for_function(args_lst)
    if is_backprop:
        data_file_name, trace_file_name = spio.benchmark.profile_function_training(
            function_reflection,
            function,
            params,
            args_lst,
            warmup=args.warmup,
            num_iters=args.num_iters,
        )
    else:
        data_file_name, trace_file_name = spio.benchmark.profile_function_inference(
            function_reflection,
            function,
            params,
            args_lst,
            warmup=args.warmup,
            num_iters=args.num_iters,
        )
    print(f"Wrote data to {data_file_name}")
    print(f"Wrote trace to {trace_file_name}")


def get_output_file_name(args):
    device = args.device
    datestamp = time.strftime("%Y%m%d_%H%M%S")
    device_name = get_formatted_device_name(device)
    arch = get_formatted_arch(device)
    if args.ssv:
        ext = ".ssv"
    else:
        ext = ".dat"
    return "__".join(["benchmark", device_name, arch, datestamp]) + ext


def randomize_batch_size(
    N: int, divisor: int = 32, min_scale: float = 0.5, max_scale: float = 2.0
) -> int:
    """Return a random batch size that is a multiple of `divisor`.

    The given batch size is scaled by a random factor between `min_scale` and `max_scale`.

    Args:
        N: The original batch size.
        divisor: The returned batch size will be a multiple of this value.
        min_scale: The minimum scaling factor.
        max_scale: The maximum scaling factor."""
    random_N = N * random.uniform(min_scale, max_scale)
    random_Nd = max(1, int(round(random_N / divisor)))
    random_N_divisible = random_Nd * divisor
    return random_N_divisible


if __name__ == "__main__":
    main()
