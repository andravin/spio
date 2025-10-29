"""Helper function for implementing kernel and operator unit tests."""

from typing import List, Type, Callable
import contextlib
import os

import torch

from ..compiler import compile_kernel_configs
from ..util.close import assert_all_close_with_acc_depth
from ..util.device_info import get_device_ordinal
from ..reflection import (
    get_kernel_reflection,
    get_function_reflection,
    get_layer_reflection,
)
from ..cuda.driver import get_device_attributes
from ..transform._transform import _transform as spio_transform
from ..kernels import Params, KernelFactory


# Pre-fill output tensors with this value to ensure every element is written by the kernel.
DEAD_VALUE = -42.0


def run_kernel_test(
    kernel_factory: KernelFactory,
    params: Params,
    device: str = "cuda",
    configs=None,
    **kernel_kwargs,
):
    """Run a test for an forward pass kernel."""
    kernel_name = kernel_factory.get_kernel_name(**kernel_kwargs)
    reflection = get_kernel_reflection(kernel_name)

    reference_function = reflection.reference
    reference_reflection = get_function_reflection(reference_function)

    args = reference_reflection.make_args(params, device=device)
    kernel_args = reflection.arrange_args(args)

    reference_args = _all_to_float(reference_reflection.arrange_args(args))
    torch_kwargs = reference_reflection.get_function_kwargs(params)

    device_idx = get_device_ordinal(device)
    device_attr = get_device_attributes(device_idx)

    with _precision_guard():
        torch_output = reference_function(*reference_args, **torch_kwargs)
    kernels = compile_kernel_configs(
        kernel_factory,
        params,
        configs=configs,
        device_attr=device_attr,
        **reflection.kwargs,
    )
    stats = reflection.stats(params, output_names=reflection.output_names)
    acc_depth = stats.accumulation_depths[0]
    output_name = reflection.output_names[0]
    output = args[output_name]
    for kernel in kernels:
        kernel.load()
        output.fill_(DEAD_VALUE)
        reflection.init_zeros(args)
        kernel(*kernel_args)
        assert_all_close_with_acc_depth(
            output, torch_output, msg=str(kernel), acc_depth=acc_depth
        )


def run_grad_kernel_test(
    kernel_factory: KernelFactory,
    params: Params,
    device: str = "cuda",
    configs=None,
    **kernel_kwargs,
):
    """Run a test for a backward pass kernel."""
    device_idx = get_device_ordinal(device)
    device_attr = get_device_attributes(device_idx)
    kernel_name = kernel_factory.get_kernel_name(**kernel_kwargs)
    reflection = get_kernel_reflection(kernel_name)
    args = reflection.make_args(params, device=device, training=True)

    reference_function = reflection.reference
    reference_reflection = get_function_reflection(reference_function)
    reference_args = _all_to_float(reference_reflection.arrange_args(args))
    reference_kwargs = reference_reflection.get_function_kwargs(params)

    grad_outputs = reflection.get_grad_output_args(args)

    with _precision_guard():
        output = reference_function(*reference_args, **reference_kwargs)
        num_grad_inputs = len(reflection.grad_input_names)
        ref_grads = {
            grad_input.name: torch.autograd.grad(
                output,
                args[grad_input.grad_of],
                grad_outputs,
                retain_graph=(idx < num_grad_inputs - 1),
            )[0]
            for idx, grad_input in enumerate(reflection.grad_input_names)
        }

    grad_input_names = [grad_name.name for grad_name in reflection.grad_input_names]
    grad_input_stats = [
        reflection.stats(params, output_names=[output_name])
        for output_name in grad_input_names
    ]
    acc_depths = [stats.accumulation_depths[0] for stats in grad_input_stats]

    kernel_args = reflection.arrange_args(args)
    kernels = compile_kernel_configs(
        kernel_factory,
        params,
        configs=configs,
        device_attr=device_attr,
        **kernel_kwargs,
    )
    for kernel in kernels:
        kernel.load()
        reflection.init_zeros(args)
        kernel(*kernel_args)
        for grad_name, acc_depth in zip(reflection.grad_input_names, acc_depths):
            grad = args[grad_name.name]
            ref_grad = ref_grads[grad_name.name]
            assert_all_close_with_acc_depth(
                grad, ref_grad, msg=str(kernel), acc_depth=acc_depth
            )


def run_opcheck_test(function, params, device="cuda"):
    """Check if a PyTorch custom operator was registered correctly."""
    reflection = get_function_reflection(function)
    args = reflection.make_args(params, device=device)
    function_args = reflection.arrange_args(args)
    function_kwargs = reflection.get_function_kwargs(params)
    torch.library.opcheck(
        function, function_args, function_kwargs, raise_exception=True
    )


def run_function_test(function, params, device="cuda"):
    """Run a test for the forward pass of a function."""
    reflection = get_function_reflection(function)

    reference_function = reflection.reference
    reference_reflection = get_function_reflection(reference_function)

    args = reference_reflection.make_args(params, device=device)
    function_args = reflection.arrange_args(args)
    function_args = _all_to_float(function_args)
    function_kwargs = reflection.get_function_kwargs(params)

    reference_args = _all_to_float(reference_reflection.arrange_args(args))
    reference_kwargs = reference_reflection.get_function_kwargs(params)

    with _precision_guard():
        reference_output = reference_function(*reference_args, **reference_kwargs)

    stats = reflection.stats(params, output_names=reflection.output_names)
    acc_depth = stats.accumulation_depths[0]

    with torch.autocast(device_type=device, dtype=torch.float16):
        output = function(*function_args, **function_kwargs)

    assert_all_close_with_acc_depth(
        output, reference_output, msg=str(params), acc_depth=acc_depth
    )


def run_grad_function_test(function: Callable, params: Params, device: str = "cuda"):
    """Run a test for a backward pass of a function."""
    reflection = get_function_reflection(function)
    args = reflection.make_args(params, device=device, training=True)
    function_args = reflection.arrange_args(args)
    function_args = _all_to_float(function_args)
    function_kwargs = reflection.get_function_kwargs(params)
    with torch.autocast(device_type=device, dtype=torch.float16):
        output = function(*function_args, **function_kwargs)

    reference_function = reflection.reference
    reference_reflection = get_function_reflection(reference_function)
    reference_args = reference_reflection.arrange_args(args)
    reference_kwargs = reference_reflection.get_function_kwargs(params)

    float_args = _all_to_float(reference_args)
    with _precision_guard():
        reference_outputs = reference_function(*float_args, **reference_kwargs)
    grad_outputs = list(reflection.make_grad_outputs(params).values())
    float_grad_outputs = _all_to_float(grad_outputs)
    grad_input_names = [f"grad_{name}" for name in reflection.args]
    grad_input_stats = [
        reflection.stats(params, output_names=[output_name])
        for output_name in grad_input_names
    ]
    acc_depths = [stats.accumulation_depths[0] for stats in grad_input_stats]
    for idx, (arg, float_arg) in enumerate(zip(function_args, float_args)):
        if arg is not None:
            not_last = idx < len(args) - 1
            grad = torch.autograd.grad(
                output, arg, *grad_outputs, retain_graph=not_last
            )
            with _precision_guard():
                reference_grad = torch.autograd.grad(
                    reference_outputs, float_arg, *float_grad_outputs, retain_graph=not_last
                )
            assert_all_close_with_acc_depth(
                grad[0], reference_grad[0], msg=str(params), acc_depth=acc_depths[idx]
            )


def run_layer_test(
    layer_cls: Type[torch.nn.Module],
    params: Params,
    device: str = "cuda",
    torchcompile: bool = False,
):
    """Run a test for a Spio layer."""
    reflection = get_layer_reflection(layer_cls)
    args = reflection.make_args(params, device=device)
    layer_args = reflection.arrange_args(args)
    layer_args = _all_to_float(layer_args)

    reference_layer_cls = reflection.reference
    reference_reflection = get_layer_reflection(reference_layer_cls)
    reference_args = reference_reflection.arrange_args(args)
    reference_kwargs = reference_reflection.get_function_kwargs(params)
    reference_args = _all_to_float(reference_args)

    reference_layer = reference_layer_cls(**reference_kwargs).to(device=device)
    with torch.no_grad():
        reference_layer.weight.copy_(args["weight"].float().clone().detach())
        if params.has_bias:
            reference_layer.bias.copy_(args["bias"].float().clone().detach())
    reference_layer = reference_layer.to(
        memory_format=reference_reflection.memory_format
    )

    reference_model = torch.nn.Sequential(reference_layer)
    with _precision_guard():
        reference_output = reference_model(*reference_args)

    layer, num_spio_modules = spio_transform(reference_model)
    assert num_spio_modules == 1, f"Expected 1 Spio module, matched {num_spio_modules}"

    if torchcompile:
        layer = torch.compile(layer)

    with torch.autocast(device_type=device, dtype=torch.float16):
        output = layer(*layer_args)

    stats = reflection.stats(params, output_names=reflection.output_names)
    acc_depth = stats.accumulation_depths[0]

    assert_all_close_with_acc_depth(
        output, reference_output, msg=str(params), acc_depth=acc_depth
    )


def _all_to_float(args: List[torch.Tensor]) -> List[torch.Tensor]:
    return [t.float() if t is not None else None for t in args]

@contextlib.contextmanager
def _precision_guard():
    # Disable TF32 control if explicitly requested
    if os.environ.get("SPIO_DISABLE_TF32", "1") == "0":
        yield
        return

    # Prefer PyTorch 2.9+ API; do not mix with old flags
    try:
        old_states = {
            "backends.fp32_precision": torch.backends.fp32_precision,
            "cuda.matmul.fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "cudnn.fp32_precision": torch.backends.cudnn.fp32_precision,
            "cudnn.conv.fp32_precision": torch.backends.cudnn.conv.fp32_precision,
            "cudnn.rnn.fp32_precision": torch.backends.cudnn.rnn.fp32_precision,
            "cudnn.benchmark": torch.backends.cudnn.benchmark,
        }
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
        torch.backends.cudnn.benchmark = False
        try:
            yield
        finally:
            torch.backends.fp32_precision = old_states["backends.fp32_precision"]
            torch.backends.cuda.matmul.fp32_precision = old_states["cuda.matmul.fp32_precision"]
            torch.backends.cudnn.fp32_precision = old_states["cudnn.fp32_precision"]
            torch.backends.cudnn.conv.fp32_precision = old_states["cudnn.conv.fp32_precision"]
            torch.backends.cudnn.rnn.fp32_precision = old_states["cudnn.rnn.fp32_precision"]
            torch.backends.cudnn.benchmark = old_states["cudnn.benchmark"]
        return
    except AttributeError:
        # Older PyTorch: fall back to legacy flags
        pass

    # Legacy path (pre-2.9)
    old_mm = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_bench = torch.backends.cudnn.benchmark
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_mm
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.backends.cudnn.benchmark = old_bench
