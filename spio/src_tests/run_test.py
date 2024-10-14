import torch

from ..compiler import compile_kernel_configs
from ..util.close import assert_all_close
from ..reflection import (
    get_kernel_reflection,
    get_function_reflection,
    get_layer_reflection,
)
from ..transform._transform import _transform as spio_transform
from ..kernels.kernel_factory import KernelFactory


def run_kernel_test(
    kernel_factory: KernelFactory, params, device="cuda", **kernel_kwargs
):
    """Run a test for an forward pass kernel."""
    arch = torch.cuda.get_device_capability(device)

    kernel_name = kernel_factory.get_kernel_name(**kernel_kwargs)
    reflection = get_kernel_reflection(kernel_name)
    args = reflection.make_args(params, device=device)
    kernel_args = reflection.arrange_args(args)

    reference_function = reflection.reference
    reference_reflection = get_function_reflection(reference_function)
    reference_args = reference_reflection.arrange_args(args)
    torch_kwargs = reference_reflection.get_function_kwargs(params)

    torch_output = reference_function(*reference_args, **torch_kwargs)
    kernels = compile_kernel_configs(
        kernel_factory, params, arch=arch, **reflection.kwargs
    )
    for kernel in kernels:
        kernel.load()
        reflection.init_zeros(args)
        kernel(*kernel_args)
        output_name = reflection.output_names[0]
        output = args[output_name]
        assert_all_close(output, torch_output, msg=str(kernel))


def run_grad_kernel_test(
    kernel_factory: KernelFactory, params, device="cuda", **kernel_kwargs
):
    """Run a test for a backward pass kernel."""
    arch = torch.cuda.get_device_capability(device)
    kernel_name = kernel_factory.get_kernel_name(**kernel_kwargs)
    reflection = get_kernel_reflection(kernel_name)
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
    kernels = compile_kernel_configs(kernel_factory, params, arch=arch, **kernel_kwargs)
    for kernel in kernels:
        kernel.load()
        reflection.init_zeros(args)
        kernel(*kernel_args)
        for grad_name in reflection.grad_input_names:
            grad = args[grad_name.name]
            ref_grad = ref_grads[grad_name.name]
            assert_all_close(grad, ref_grad, msg=str(kernel))


def run_opcheck_test(function, params, device="cuda"):
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
    """Run a test for a backward pass of a function."""
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


def run_layer_test(
    layer_cls, params, device="cuda", torchcompile=False, torchcompile_mode=None
):
    """Run a test for a torch.nn.Module layer subclass that uses Spio a function."""
    reflection = get_layer_reflection(layer_cls)
    args = reflection.make_args(params, device=device)
    layer_args = reflection.arrange_args(args)

    reference_layer_cls = reflection.reference
    reference_reflection = get_layer_reflection(reference_layer_cls)
    reference_args = reference_reflection.arrange_args(args)
    reference_kwargs = reference_reflection.get_function_kwargs(params)

    # Let's force all args to float to test autocast.
    reference_args = [a.float() if a is not None else None for a in reference_args]

    reference_layer = reference_layer_cls(**reference_kwargs).to(
        device=device, memory_format=reference_reflection.memory_format
    )

    reference_model = torch.nn.Sequential(reference_layer)

    layer, num_spio_modules = spio_transform(reference_model)
    assert num_spio_modules == 1, f"Expected 1 Spio module, matched {num_spio_modules}"

    if torchcompile:
        layer = torch.compile(layer, mode=torchcompile_mode)

    with torch.autocast(device_type=device, dtype=torch.float16):
        output = layer(*layer_args)

    with torch.autocast(device_type=device, dtype=torch.float16):
        reference_output = reference_model(*reference_args)

    assert_all_close(output, reference_output, msg=str(params))
