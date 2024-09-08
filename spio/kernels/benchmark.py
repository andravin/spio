from typing import Any, List
from dataclasses import dataclass
import inspect

import torch

from ..compiler import compile_kernel_configs
from .benchmark_result import BenchmarkResult
from spio import cuda


def _benchmark_kernel(
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


def benchmark(
    kernel_cls,
    params,
    kernel_args_lst,
    configs=None,
    warmup=10,
    num_iters=10,
    device="cuda",
    **kernel_kwargs,
) -> BenchmarkResult:
    with torch.device(device) as device_obj:
        device_ordinal = device_obj.index if device_obj.index is not None else 0
        cuda.primary_context_guard.set_device(device_ordinal)
        arch = torch.cuda.get_device_capability(device=device_obj)
        if configs is None:
            configs = list(kernel_cls.configs(params))
        kernels = compile_kernel_configs(
            kernel_cls, params, configs=configs, arch=arch, **kernel_kwargs
        )
        num_args = len(kernel_args_lst)
        best_result = BenchmarkResult(time_ms=float("inf"))
        best_kernel = None
        device_name = torch.cuda.get_device_name(device_obj)
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
                api_version = cuda.primary_context_guard.get_api_version()
                driver_version = cuda.driver.get_driver_version()
                raise ValueError(
                    f"Error loading kernel {kernel} with config {config}: {e}; API version: {api_version} Driver version: {driver_version}"
                ) from e

            time_ms = _benchmark_kernel(
                kernel, kernel_args_lst, warmup=warmup, num_iters=num_iters
            )
            result = BenchmarkResult(
                kernel_cls=kernel_cls,
                device_desc=device_name,
                params=params,
                config=config,
                kernel_idx=idx,
                kernel_kwargs=kernel_kwargs,
                time_ms=time_ms,
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


def get_full_name(obj):
    return f"{obj.__module__}.{obj.__name__}"


def benchmark_function(
    kernel_cls, function, params, function_args_lst, warmup=10, num_iters=10
):
    name = get_full_name(function)
    time_ms = _benchmark_function(
        function,
        function_args_lst,
        warmup=warmup,
        num_iters=num_iters,
        function_kwargs=params.kwargs,
    )
    return BenchmarkResult(
        time_ms=time_ms, kernel_cls=kernel_cls, params=params, name=function.__name__
    )


def _benchmark_function(
    function, args_lst, warmup: int = 10, num_iters: int = 10, function_kwargs={}
) -> float:
    total_iters = warmup + num_iters
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    num_args = len(args_lst)
    x = args_lst[0].pop(0)
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


def benchmark_grad_reference(
    reflection,
    kernel_cls,
    params,
    warmup=10,
    num_iters=10,
    device="cuda",
):
    time_ms = _benchmark_grad_reference(
        reflection,
        params,
        warmup=warmup,
        num_iters=num_iters,
        device=device,
    )
    return BenchmarkResult(
        time_ms=time_ms,
        kernel_cls=kernel_cls,
        params=params,
        name=f"{reflection.reference.__name__}_grad",
    )


def _benchmark_grad_reference(
    reflection, params, warmup=10, num_iters=10, device="cuda"
):
    name = get_full_name(reflection.reference)
    grad_input_name_str = "_".join([grad.name for grad in reflection.grad_input_names])
    args = reflection.make_args(params, device=device, training=True)
    reference_args = reflection.arrange_reference_args(args)
    grad_outputs = reflection.get_grad_output_args(args)
    grad_inputs = reflection.get_differentiable_input_args(args)
    function = reflection.reference
    output = function(*reference_args, **params.kwargs)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_iters = warmup + num_iters
    for i in range(warmup):
        for grad_input in grad_inputs:
            grad = torch.autograd.grad(
                output, grad_input, grad_outputs, retain_graph=True
            )
    start_event.record()
    for i in range(warmup, total_iters):
        for gi, grad_input in enumerate(grad_inputs):
            not_last = (i < total_iters - 1) or (gi < len(grad_inputs) - 1)
            grad = torch.autograd.grad(
                output, grad_input, grad_outputs, retain_graph=not_last
            )
    end_event.record()
    end_event.synchronize()
    time_ms = start_event.elapsed_time(end_event)
    iter_time_ms = time_ms / num_iters
    return iter_time_ms
