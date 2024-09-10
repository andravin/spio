import torch

from ..log import BenchmarkResult
from ..cuda import primary_context_guard, driver
from ..compiler import compile_kernel_configs
from ..util import get_formatted_device_name, get_formatted_arch
from .stats import Stats


def _benchmark_kernel(
    kernel,
    args_lst,
    warmup: int = 10,
    num_iters: int = 10,
    prefix_op=None,
    kernel_kwargs={},
) -> float:
    total_iters = warmup + num_iters
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    num_args = len(args_lst)
    for i in range(warmup):
        kernel(*args_lst[i % num_args], **kernel_kwargs)
    if prefix_op is not None:
        prefix_op(args_lst[-1][1])  # TODO Assume the second arg is the input.
    start_event.record()
    for i in range(warmup, total_iters):
        kernel(*args_lst[i % num_args], **kernel_kwargs)
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    time_ms = start_event.elapsed_time(end_event)
    iter_time_ms = time_ms / num_iters
    return iter_time_ms


def benchmark_kernel(
    kernel_cls,
    params,
    kernel_args_lst,
    configs=None,
    warmup=10,
    num_iters=10,
    device="cuda",
    prefix_op=None,
    **kernel_kwargs,
) -> BenchmarkResult:
    with torch.device(device) as device_obj:
        device_ordinal = device_obj.index if device_obj.index is not None else 0
        device_name = get_formatted_device_name(device)
        arch = torch.cuda.get_device_capability(device=device_obj)
        arch_str = get_formatted_arch(device)
        primary_context_guard.set_device(device_ordinal)
        if configs is None:
            configs = list(kernel_cls.configs(params))
        kernels = compile_kernel_configs(
            kernel_cls, params, configs=configs, arch=arch, **kernel_kwargs
        )
        best_result = BenchmarkResult(time_ms=float("inf"))
        best_kernel = None
        results = []
        for idx, (kernel, config) in enumerate(zip(kernels, configs)):
            try:
                # FIXME Unsafe to load kernels while cuda is running?
                # FIXME but what is running?
                # FIXME is there some garbage collection that torch is doing?
                # vvvvvvvvvvvvvvvvvvvvvv
                torch.cuda.synchronize()
                # ^^^^^^^^^^^^^^^^^^^^^^
                # FIXME race condition here that sometimes causes CUDA_ERROR_INVALID_CONTEXT.
                # FIXME the above syncronization is a hack to fix it.
                kernel.load(device_ordinal=device_ordinal)
            except ValueError as e:
                api_version = primary_context_guard.get_api_version()
                driver_version = driver.get_driver_version()
                raise ValueError(
                    f"Error loading kernel {kernel} with config {config}: {e}; API version: {api_version} Driver version: {driver_version}"
                ) from e

            time_ms = _benchmark_kernel(
                kernel,
                kernel_args_lst,
                warmup=warmup,
                num_iters=num_iters,
                prefix_op=prefix_op,
            )
            result = BenchmarkResult(
                name=kernel.kernel_name,
                device_desc=device_name,
                arch=arch_str,
                params=params,
                config=config,
                kernel_idx=idx,
                kernel_kwargs=kernel_kwargs,
                time_ms=time_ms,
                stats=kernel.stats,
            )
            if result.time_ms < best_result.time_ms:
                if best_kernel is not None:
                    best_kernel.unload()
                best_result = result
                best_kernel = kernel
            else:
                kernel.unload()
            results.append(result)
        return (best_result, best_kernel, results)
