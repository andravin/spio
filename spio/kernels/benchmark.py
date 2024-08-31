from typing import Any
from dataclasses import dataclass
import inspect

import torch

from ..compiler import compile_kernels


def benchmark_kernel(
    kernel, args_lst, warmup: int = 10, num_iters: int = 10, kernel_kwargs={}
) -> float:
    total_iters = warmup + num_iters
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    num_args = len(args_lst)
    for i in range(warmup):
        kernel(*args_lst[i % num_args], **kernel_kwargs)
    start_event.record()
    for i in range(warmup, total_iters):
        kernel(*args_lst[i % num_args], **kernel_kwargs)
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    time_ms = start_event.elapsed_time(end_event)
    iter_time_ms = time_ms / num_iters
    return iter_time_ms


def benchmark_function(
    function, input, args_lst, warmup: int = 10, num_iters: int = 10, function_kwargs={}
) -> float:
    total_iters = warmup + num_iters
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    num_args = len(args_lst)
    x = input
    for i in range(warmup):
        x = function(x, *args_lst[i % num_args], **function_kwargs)
    start_event.record()
    for i in range(warmup, total_iters):
        x = function(x, *args_lst[i % num_args], **function_kwargs)
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    time_ms = start_event.elapsed_time(end_event)
    iter_time_ms = time_ms / num_iters
    return iter_time_ms


@dataclass
class BenchmarkResult:
    config: Any = None
    kernel_idx: int = None
    kernel: Any = None
    time_ms: float = None
    tflop_s: float = None
    eff_bw_gb_s: float = None

    @property
    def time_s(self):
        return self.time_ms / 1e3

    @staticmethod
    def header():
        config = "Config"
        time = "Time[ms]"
        tflop_s = "TFLOP/s"
        eff_bw_gb_s = "Eff BW[GB/s]"
        idx = "Idx"
        print(f"[{idx:4s}] {config:70s} {time:8s} {tflop_s:8s} {eff_bw_gb_s:8s}")

    def print(self, best=False):
        idx = str(self.kernel_idx)
        if best:
            idx = f"{idx} *"
        print(
            f"[{idx:4s}] {str(self.config):70s} {self.time_ms:8.3f} {self.tflop_s:8.3f} {self.eff_bw_gb_s:8.3f}"
        )

    def calculate_performance(self, gmacs: int, gbytes: int):
        self.tflop_s = gmacs / self.time_ms
        self.eff_bw_gb_s = gbytes / self.time_s


def benchmark(
    kernel_cls, params, kernel_args_lst, configs=None, warmup=10, num_iters=10, **kwargs
) -> BenchmarkResult:
    best = BenchmarkResult(time_ms=float("inf"))
    print(f"{kernel_cls.__name__}{kwargs} {params}")
    if configs is None:
        configs = list(kernel_cls.configs(params))
    kernels = [kernel_cls(params, config, **kwargs) for config in configs]
    compiler_args = [kernel.compiler_args for kernel in kernels]
    results = compile_kernels(compiler_args)
    bytes = kernel_cls.bytes_read(params, **kwargs) + kernel_cls.bytes_written(
        params, **kwargs
    )
    macs = kernel_cls.macs(params, **kwargs)
    gmacs = macs / 1e9
    gbytes = bytes / 1e9

    num_args = len(kernel_args_lst)

    BenchmarkResult.header()
    for idx, (kernel, config) in enumerate(zip(kernels, configs)):
        kernel.load()
        r = BenchmarkResult(config=config, kernel_idx=idx, kernel=kernel)
        r.time_ms = benchmark_kernel(
            kernel, kernel_args_lst, warmup=warmup, num_iters=num_iters
        )
        r.calculate_performance(gmacs, gbytes)
        if r.time_ms < best.time_ms:
            best = r
        r.print()
    if len(kernels) > 1:
        best.print(best=True)
    return best


def get_full_name(obj):
    return f"{obj.__module__}.{obj.__name__}"


def benchmark_reference(
    kernel_cls, params, input, torch_args_lst, warmup=10, num_iters=10
):
    function = params.reference
    name = get_full_name(function)
    print(f"{name} {params}")
    bytes = kernel_cls.bytes_read(params) + kernel_cls.bytes_written(params)
    macs = kernel_cls.macs(params)
    gmacs = macs / 1e9
    gbytes = bytes / 1e9
    BenchmarkResult.header()
    time_ms = benchmark_function(
        params.reference,
        input,
        torch_args_lst,
        warmup=warmup,
        num_iters=num_iters,
        function_kwargs=params.kwargs,
    )
    result = BenchmarkResult(time_ms=time_ms)
    result.calculate_performance(gmacs, gbytes)
    result.print()
    return result
