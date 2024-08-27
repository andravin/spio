import torch

from .close import assert_all_close
from .compiler_pool import compile_kernels


def run_kernel_test(kernel_cls, params):
    inputs = params.random_inputs()
    weights = params.random_weights()
    torch_args = inputs + weights
    torch_outputs = params.reference(*torch_args, **params.kwargs)
    outputs = params.empty_outputs()
    kernel_args = kernel_cls.arrange_kernel_args(
        inputs=inputs, weights=weights, outputs=outputs
    )
    kernels = [
        kernel_cls.fprop_kernel(params, kernel_args, config=config)
        for config in kernel_cls.configs(params)
    ]
    compiler_args = [kernel.compiler_args for kernel in kernels]
    compile_kernels(compiler_args)
    for kernel in kernels:
        kernel.load()
        kernel(*kernel_args)
        output = outputs[0]
        assert_all_close(output, torch_outputs, msg=str(params))


def run_function_test(function, params):
    args = params.random_args()
    outputs = function(*args, **params.kwargs)
    float_args = [a.float() if a is not None else None for a in args]
    torch_outputs = params.reference(*float_args, **params.kwargs)
    assert_all_close(outputs, torch_outputs, msg=str(params))


def run_function_tests(function, test_params):
    for params in test_params:
        run_function_test(function, params)


def run_kernel_tests(kernel_cls, test_params):
    for params in test_params:
        run_kernel_test(kernel_cls, params)


def run_grad_function_test(function, params):
    args = params.random_args(training=True)
    float_args = [a.float() if a is not None else None for a in args]
    kwargs = params.kwargs
    output = function(*args, **kwargs)
    torch_outputs = params.reference(*float_args, **kwargs)
    deltas = params.random_deltas(output.device)
    float_deltas = [delta.float() for delta in deltas]
    for idx, (arg, float_arg) in enumerate(zip(args, float_args)):
        if arg is not None:
            not_last = idx < len(args) - 1
            grad = torch.autograd.grad(output, arg, *deltas, retain_graph=not_last)
            torch_grad = torch.autograd.grad(
                torch_outputs, float_arg, *float_deltas, retain_graph=not_last
            )
            assert_all_close(grad[0], torch_grad[0], msg=str(params))


def run_grad_function_tests(function, test_params):
    for params in test_params:
        run_grad_function_test(function, params)
