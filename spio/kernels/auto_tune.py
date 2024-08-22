import torch

from ..compiler_pool import compile_kernels


def benchmark_kernel(
    kernel,
    args,
    warmup: int = 10,
    num_iters: int = 10,
) -> float:
    total_iters = warmup + num_iters
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(warmup):
        kernel(*args)
    start_event.record()
    for i in range(warmup, total_iters):
        kernel(*args)
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    time_ms = start_event.elapsed_time(end_event)
    iter_time_ms = time_ms / num_iters
    return iter_time_ms


def auto_tune(kernel_cls, params, args, warmup=10, num_iters=10, **kwargs):
    best_time = float("inf")
    best_kernel = None
    best_idx = None
    best_config = None
    print(f"Auto-tuning {params} {kwargs}")
    configs = list(kernel_cls.configs(params))
    kernels = [kernel_cls(params, config, **kwargs) for config in configs]
    compiler_args = [kernel.compiler_args for kernel in kernels]
    results = compile_kernels(compiler_args)
    for idx, (kernel, config) in enumerate(zip(kernels, configs)):
        kernel.load()
        time = benchmark_kernel(kernel, args, warmup=warmup, num_iters=num_iters)
        print(f"config[{idx:2}]: {str(config):70s} {time:>5.3f} ms")
        if time < best_time:
            best_time = time
            best_idx = idx
            best_kernel = kernel
            best_config = config
    print(f"* best[{best_idx:2}]: {str(best_config):70s} {best_time:>5.3f} ms")
    return best_kernel, best_idx, best_time
