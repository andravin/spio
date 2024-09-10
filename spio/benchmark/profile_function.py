from typing import Any, List
from dataclasses import dataclass
from datetime import datetime

import torch

from ..reflection import get_function_reflection
from ..kernels import Stats
from ..util import IntervalTimer


def profile_function_training(
    reflection,
    function,
    params,
    args_lst,
    warmup=10,
    num_iters=1,
    base_file_name="profile",
):
    function = reflection.function
    function_kwargs = reflection.get_function_kwargs(params)
    wait_iters = 1
    total_iters = warmup + num_iters + wait_iters
    depth = len(args_lst)
    first_x = args_lst[0].pop(0)
    c = first_x.shape[1]
    prefix = torch.nn.Conv2d(c, c, kernel_size=3, padding=1).cuda()
    postfix = torch.nn.Conv2d(c, c, kernel_size=3, padding=1).cuda()

    schedule = torch.profiler.schedule(
        wait=wait_iters, warmup=warmup, active=num_iters, repeat=1
    )
    with torch.profiler.profile(schedule=schedule) as prof:
        for i in range(total_iters):
            x = first_x
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                x = prefix(x)
                for i in range(depth):
                    x = function(x, *args_lst[i], **function_kwargs)
                x = postfix(x).sum()
            x.backward()
            # This synchronization causes the steps of the profile to line up with
            # kernel executions. Use a large prefix op and sufficient depth
            # to ensure execution time is not dominated by launch overhead.
            torch.cuda.synchronize()
            prof.step()

    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    name = _get_full_name_with_underscores(function)
    trace_file_name = _get_profile_file_name(name, datestamp, prefix="train")
    data_file_name = _get_profile_file_name(name, datestamp, ext=".dat", prefix="train")
    prof.export_chrome_trace(trace_file_name)
    with open(data_file_name, "w") as f:
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        f.write(table)

    return data_file_name, trace_file_name


def profile_function_inference(
    reflection,
    function,
    params,
    args_lst,
    warmup=10,
    num_iters=1,
    base_file_name="profile",
):
    function = reflection.function
    function_kwargs = reflection.get_function_kwargs(params)
    wait_iters = 1
    total_iters = warmup + num_iters + wait_iters
    depth = len(args_lst)
    first_x = args_lst[0].pop(0)
    c = first_x.shape[1]
    prefix = torch.nn.Conv2d(c, c, kernel_size=3, padding=1).cuda()
    postfix = torch.nn.Conv2d(c, c, kernel_size=3, padding=1).cuda()

    schedule = torch.profiler.schedule(
        wait=wait_iters, warmup=warmup, active=num_iters, repeat=1
    )
    with torch.profiler.profile(schedule=schedule) as prof:
        for i in range(total_iters):
            with torch.no_grad():
                x = first_x
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    x = prefix(x)
                    for i in range(depth):
                        x = function(x, *args_lst[i], **function_kwargs)
                    x = postfix(x).sum()
            # This synchronization causes the steps of the profile to line up with
            # kernel executions. Use a large prefix op and sufficient depth
            # to ensure execution time is not dominated by launch overhead.
            torch.cuda.synchronize()
            prof.step()

    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = _get_full_name_with_underscores(function)
    trace_file_name = _get_profile_file_name(name, datestamp, prefix="inference")
    data_file_name = _get_profile_file_name(
        name, datestamp, ext=".dat", prefix="inference"
    )
    prof.export_chrome_trace(trace_file_name)
    with open(data_file_name, "w") as f:
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        f.write(table)

    return data_file_name, trace_file_name


def _get_full_name_with_underscores(obj):
    full_name = _get_full_name(obj)
    return full_name.replace(".", "_")


def _get_full_name(obj):
    return f"{obj.__module__}.{obj.__name__}"


def _get_profile_file_name(
    base_file_name: str, datestamp: str, ext: str = ".json", prefix="profile"
) -> str:
    return f"{prefix}_{base_file_name}_{datestamp}{ext}"
