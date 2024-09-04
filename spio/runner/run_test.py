import torch

from ..compiler import compile_kernel_configs
from ..util.close import assert_all_close
from ..reflection import get_kernel_reflection, get_function_reflection


def run_kernel_test(kernel_cls, params, **kernel_kwargs):
    reflection = get_kernel_reflection(kernel_cls, **kernel_kwargs)
    args = reflection.make_args(params)
    kernel_args = reflection.arrange_kernel_args(args)
    reference_args = reflection.arrange_reference_args(args)
    torch_output = reflection.reference(*reference_args, **params.kwargs)
    kernels = compile_kernel_configs(kernel_cls, params, **reflection.kernel_kwargs)
    for kernel in kernels:
        kernel.load()
        reflection.zero_args(args)
        kernel(*kernel_args)
        output_name = reflection.output_names[0]
        output = args[output_name]
        assert_all_close(output, torch_output, msg=str(kernel))


def run_grad_kernel_test(kernel_cls, params, **kernel_kwargs):
    reflection = get_kernel_reflection(kernel_cls, **kernel_kwargs)
    args = reflection.make_args(params, training=True)
    reference_args = reflection.arrange_reference_args(args)
    grad_outputs = reflection.get_grad_output_args(args)
    function = reflection.reference
    output = function(*reference_args, **params.kwargs)
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

    kernel_args = [
        arg.detach()
        for arg in reflection.arrange_kernel_args(args)
        if isinstance(arg, torch.Tensor)
    ]
    kernels = compile_kernel_configs(kernel_cls, params, **kernel_kwargs)
    for kernel in kernels:
        kernel.load()
        reflection.zero_args(args)
        kernel(*kernel_args)
        for grad_name in reflection.grad_input_names:
            grad = args[grad_name.name]
            ref_grad = ref_grads[grad_name.name]
            assert_all_close(grad, ref_grad, msg=str(kernel))


def run_function_test(function, params):
    reflection = get_function_reflection(function)
    args = reflection.make_args(params)
    function_args = reflection.arrange_function_args(args)
    reference_args = reflection.arrange_reference_args(args)
    output = function(*function_args, **params.kwargs)
    float_args = [a.float() if a is not None else None for a in reference_args]
    torch_outputs = reflection.reference(*float_args, **params.kwargs)
    assert_all_close(output, torch_outputs, msg=str(params))


def run_grad_function_test(function, params):
    reflection = get_function_reflection(function)
    args = reflection.make_args(params, training=True)
    function_args = reflection.arrange_function_args(args)
    reference_args = reflection.arrange_reference_args(args)
    output = function(*function_args, **params.kwargs)
    float_args = [a.float() if a is not None else None for a in reference_args]
    torch_outputs = reflection.reference(*float_args, **params.kwargs)
    deltas = reflection.make_deltas(params).values()
    float_deltas = [delta.float() for delta in deltas]
    for idx, (arg, float_arg) in enumerate(zip(function_args, float_args)):
        if arg is not None:
            not_last = idx < len(args) - 1
            grad = torch.autograd.grad(output, arg, *deltas, retain_graph=not_last)
            torch_grad = torch.autograd.grad(
                torch_outputs, float_arg, *float_deltas, retain_graph=not_last
            )
            assert_all_close(grad[0], torch_grad[0], msg=str(params))
