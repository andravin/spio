import torch

from ..compiler import compile_kernel_configs
from ..util.close import assert_all_close
from ..reflection import (
    get_kernel_reflection,
    get_function_reflection,
    get_layer_reflection,
)


def run_kernel_test(kernel_cls, params, device="cuda", **kernel_kwargs):
    arch = torch.cuda.get_device_capability(device)

    reflection = get_kernel_reflection(kernel_cls, **kernel_kwargs)
    args = reflection.make_args(params, device=device)
    kernel_args = reflection.arrange_args(args)

    reference_function = reflection.reference
    reference_reflection = get_function_reflection(reference_function)
    reference_args = reference_reflection.arrange_args(args)
    torch_kwargs = reference_reflection.get_function_kwargs(params)

    torch_output = reference_function(*reference_args, **torch_kwargs)
    kernels = compile_kernel_configs(kernel_cls, params, arch=arch, **reflection.kwargs)
    for kernel in kernels:
        kernel.load()
        reflection.init_zeros(args)
        kernel(*kernel_args)
        output_name = reflection.output_names[0]
        output = args[output_name]
        assert_all_close(output, torch_output, msg=str(kernel))


def run_grad_kernel_test(kernel_cls, params, device="cuda", **kernel_kwargs):
    arch = torch.cuda.get_device_capability(device)
    reflection = get_kernel_reflection(kernel_cls, **kernel_kwargs)
    args = reflection.make_args(params, device=device, training=True)

    reference_function = reflection.reference
    reference_reflection = get_function_reflection(reference_function)
    reference_args = reference_reflection.arrange_args(args)
    reference_kwargs = reference_reflection.get_function_kwargs(params)

    grad_outputs = reflection.get_grad_output_args(args)

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

    kernel_args = reflection.arrange_args(args)
    kernels = compile_kernel_configs(kernel_cls, params, arch=arch, **kernel_kwargs)
    for kernel in kernels:
        kernel.load()
        reflection.init_zeros(args)
        kernel(*kernel_args)
        for grad_name in reflection.grad_input_names:
            grad = args[grad_name.name]
            ref_grad = ref_grads[grad_name.name]
            assert_all_close(grad, ref_grad, msg=str(kernel))


def run_function_test(function, params, device="cuda"):
    reflection = get_function_reflection(function)
    args = reflection.make_args(params, device=device)
    function_args = reflection.arrange_args(args)
    function_kwargs = reflection.get_function_kwargs(params)

    reference_function = reflection.reference
    reference_reflection = get_function_reflection(reference_function)
    reference_args = reference_reflection.arrange_args(args)
    reference_kwargs = reference_reflection.get_function_kwargs(params)

    output = function(*function_args, **function_kwargs)
    float_args = [a.float() if a is not None else None for a in reference_args]
    reference_output = reference_function(*float_args, **reference_kwargs)
    assert_all_close(output, reference_output, msg=str(params))


def run_grad_function_test(function, params, device="cuda"):
    reflection = get_function_reflection(function)
    args = reflection.make_args(params, device=device, training=True)
    function_args = reflection.arrange_args(args)
    function_kwargs = reflection.get_function_kwargs(params)

    reference_function = reflection.reference
    reference_reflection = get_function_reflection(reference_function)
    reference_args = reference_reflection.arrange_args(args)
    reference_kwargs = reference_reflection.get_function_kwargs(params)

    output = function(*function_args, **function_kwargs)
    float_args = [a.float() if a is not None else None for a in reference_args]
    reference_outputs = reference_function(*float_args, **reference_kwargs)
    grad_outputs = list(reflection.make_grad_outputs(params).values())
    float_grad_outputs = [grad_output.float() for grad_output in grad_outputs]
    for idx, (arg, float_arg) in enumerate(zip(function_args, float_args)):
        if arg is not None:
            not_last = idx < len(args) - 1
            grad = torch.autograd.grad(
                output, arg, *grad_outputs, retain_graph=not_last
            )
            reference_grad = torch.autograd.grad(
                reference_outputs, float_arg, *float_grad_outputs, retain_graph=not_last
            )
            assert_all_close(grad[0], reference_grad[0], msg=str(params))


def run_layer_test(layer_cls, params, device="cuda"):
    reflection = get_layer_reflection(layer_cls)
    args = reflection.make_args(params, device=device)
    layer_args = reflection.arrange_args(args)
    layer_kwargs = reflection.get_function_kwargs(params)
    dtype = layer_args[0].dtype

    reference_layer = reflection.reference
    reference_reflection = get_layer_reflection(reference_layer)
    reference_args = reference_reflection.arrange_args(args)
    reference_kwargs = reference_reflection.get_function_kwargs(params)

    reference_layer = reference_layer(**reference_kwargs).to(
        device=device, memory_format=reference_reflection.memory_format, dtype=dtype
    )

    layer = reflection.from_layer(reference_layer).to(
        device=device, memory_format=reflection.memory_format
    )

    output = layer(*layer_args)
    reference_output = reference_layer(*reference_args)
    assert_all_close(output, reference_output, msg=str(params))
