import torch

from spio.kernels import Conv2dGw8Kernel, Conv2dGw8WgradKernel
from spio.functional import conv2d_gw8

from .reflection import KernelReflection, ArgInfo, register_reflection


def register_conv2d_gw8_reflections():
    register_reflection(
        KernelReflection(
            kernel_cls=Conv2dGw8Kernel,
            function=conv2d_gw8,
            arginfo=dict(
                output=ArgInfo(torch.float16, output=True, empty=True),
                input=ArgInfo(torch.float16, requires_grad=True),
                weight=ArgInfo(torch.float16, requires_grad=True),
            ),
            kernel_args=["output", "input", "weight"],
            function_args=["input", "weight"],
            reference=torch.nn.functional.conv2d,
            reference_args=["input", "weight"],
            stacking=[("output", "input")],
        )
    )
    register_reflection(
        KernelReflection(
            kernel_cls=Conv2dGw8WgradKernel,
            is_grad=True,
            function=None,
            arginfo=dict(
                input=ArgInfo(torch.float16, requires_grad=True),
                weight=ArgInfo(torch.float16, requires_grad=True),
                output=ArgInfo(torch.float16, output=True, empty=True),
                deltas=ArgInfo(torch.float16, grad_of="output"),
                wgrad=ArgInfo(torch.float16, zero=True, grad_of="weight"),
            ),
            kernel_args=["wgrad", "input", "deltas"],
            function_args=None,
            reference=torch.nn.functional.conv2d,
            reference_args=["input", "weight"],
        ),
    )
    register_reflection(
        KernelReflection(
            kernel_cls=Conv2dGw8Kernel,
            kernel_kwargs=dict(igrad=True),
            is_grad=True,
            function=None,
            arginfo=dict(
                input=ArgInfo(torch.float16, requires_grad=True),
                weight=ArgInfo(torch.float16, requires_grad=True),
                output=ArgInfo(torch.float16, output=True, empty=True),
                deltas=ArgInfo(torch.float16, grad_of="output"),
                igrad=ArgInfo(torch.float16, empty=True, grad_of="input"),
            ),
            kernel_args=["igrad", "deltas", "weight"],
            reference=torch.nn.functional.conv2d,
            reference_args=["input", "weight"],
        )
    )
