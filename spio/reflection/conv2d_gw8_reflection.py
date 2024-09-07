import torch

from spio.kernels import Conv2dGw8Kernel, Conv2dGw8WgradKernel
from spio.functional import conv2d_gw8

from .reflection import KernelReflection, ArgInfo, register_reflection, Init


def register_conv2d_gw8_reflections():
    register_reflection(
        KernelReflection(
            kernel_cls=Conv2dGw8Kernel,
            function=conv2d_gw8,
            arginfo=dict(
                output=ArgInfo(torch.float16, output=True, init=Init.EMPTY),
                input=ArgInfo(torch.float16, requires_grad=True),
                weight=ArgInfo(torch.float16, requires_grad=True),
                bias=ArgInfo(torch.float16, requires_grad=True),
            ),
            kernel_args=["output", "input", "weight", "bias"],
            function_args=["input", "weight", "bias"],
            reference=torch.nn.functional.conv2d,
            reference_args=["input", "weight", "bias"],
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
                bias=ArgInfo(torch.float16, requires_grad=True),
                output=ArgInfo(torch.float16, output=True, init=Init.EMPTY),
                grad_output=ArgInfo(torch.float16, grad_of="output"),
                grad_weight=ArgInfo(torch.float16, init=Init.ZERO, grad_of="weight"),
            ),
            kernel_args=["grad_weight", "input", "grad_output"],
            function_args=None,
            reference=torch.nn.functional.conv2d,
            reference_args=["input", "weight", "bias"],
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
                bias=ArgInfo(torch.float16, requires_grad=True),
                output=ArgInfo(torch.float16, output=True, init=Init.EMPTY),
                grad_output=ArgInfo(torch.float16, grad_of="output"),
                grad_input=ArgInfo(torch.float16, init=Init.EMPTY, grad_of="input"),
                none=ArgInfo(torch.float16, init=Init.NONE),
            ),
            kernel_args=["grad_input", "grad_output", "weight", "none"],
            reference=torch.nn.functional.conv2d,
            reference_args=["input", "weight", "bias"],
        )
    )
