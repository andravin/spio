"""Tests for the spio::WarpSemaphore class."""
import torch

from spio.compiler import compile_and_load_kernel


def test_warp_semaphore():
    """Test the spio::WaroSemaphore class."""
    _, semaphore_kernel = compile_and_load_kernel(
        kernel_name="warp_semaphore",
        source_file_name="semaphore.cu",
        src_module="spio.src_tests",
    )

    blocks = 1
    warps = 16
    iters_per_warp = 64
    num_iters = iters_per_warp * warps
    num_events = num_iters * 2
    event_types = torch.zeros((num_events,), dtype=torch.uint64, device="cuda")
    event_times = torch.zeros((num_events,), dtype=torch.uint64, device="cuda")

    threads = warps * 32
    grid = (blocks, 1, 1)
    block = (threads, 1, 1)

    semaphore_kernel.launch(grid, block, (event_types, event_times))

    events = [(int(time), int(typ)) for typ, time in zip(event_types, event_times)]

    max_count = 8

    count = 0
    for event in sorted(events):
        if event[1] == 1:
            count += 1
        else:
            count -= 1
        assert count <= max_count
