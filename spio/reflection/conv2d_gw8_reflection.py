import math

import torch

from ..kernels.conv2d_gw8_kernel import Conv2dGw8Kernel
from ..kernels.conv2d_gw8_wgrad_kernel import Conv2dGw8WgradKernel
from ..kernels.conv2d_stats import Conv2dStats
from ..kernels.conv2d_gw8_params import Conv2dGw8Params
from ..functional.conv_gw8_function import conv2d_gw8
from ..layers.conv_gw8 import Conv2dGw8

from .reflection import Reflection, ArgInfo, register_reflection, Init


def register_conv2d_gw8_reflections():

    # ---------------------------------------------------------------------------------------------

    conv2d_arg_info = dict(
        output=ArgInfo(dtype=torch.float16, output=True, init=Init.EMPTY),
        input=ArgInfo(dtype=torch.float16, requires_grad=True),
        weight=ArgInfo(dtype=torch.float16, requires_grad=True),
        bias=ArgInfo(dtype=torch.float16, requires_grad=True),
    )

    register_reflection(
        Reflection(
            kernel_cls=Conv2dGw8Kernel,
            arginfo=conv2d_arg_info,
            args=["output", "input", "weight", "bias"],
            kernel_outputs=["output"],
            reference=torch.nn.functional.conv2d,
            stacking=[("output", "input")],
            stats=Conv2dStats,
            prefix_op_constructor=_make_prefix_op,
        )
    )

    register_reflection(
        Reflection(
            function=conv2d_gw8,
            arginfo=conv2d_arg_info,
            args=["input", "weight", "bias"],
            reference=torch.nn.functional.conv2d,
            stacking=[("output", "input")],
            stats=Conv2dStats,
            params=Conv2dGw8Params,
            prefix_op_constructor=_make_prefix_op,
            get_function_kwargs=_spio_conv2d_gw8_kwargs,
        )
    )

    register_reflection(
        Reflection(
            function=torch.nn.functional.conv2d,
            arginfo=conv2d_arg_info,
            args=["input", "weight", "bias"],
            reference=None,
            stacking=[("output", "input")],
            stats=Conv2dStats,
            params=Conv2dGw8Params,
            get_function_kwargs=_torch_conv2d_kwargs,
        )
    )

    register_reflection(
        Reflection(
            layer_cls=Conv2dGw8,
            arginfo=conv2d_arg_info,
            args=["input"],
            reference=torch.nn.Conv2d,
            stats=Conv2dStats,
            params=Conv2dGw8Params,
            prefix_op_constructor=_make_prefix_op,
            get_function_kwargs=_conv2d_layer_kwargs,
            from_layer=Conv2dGw8.from_torch_module,
        )
    )

    register_reflection(
        Reflection(
            layer_cls=torch.nn.Conv2d,
            arginfo=conv2d_arg_info,
            args=["input"],
            stats=Conv2dStats,
            params=Conv2dGw8Params,
            prefix_op_constructor=_make_prefix_op,
            get_function_kwargs=_conv2d_layer_kwargs,
        )
    )

    # ---------------------------------------------------------------------------------------------

    register_reflection(
        Reflection(
            kernel_cls=Conv2dGw8WgradKernel,
            arginfo=dict(
                input=ArgInfo(dtype=torch.float16, requires_grad=True),
                weight=ArgInfo(dtype=torch.float16, requires_grad=True),
                bias=ArgInfo(dtype=torch.float16, requires_grad=True),
                output=ArgInfo(dtype=torch.float16, output=True, init=Init.EMPTY),
                grad_output=ArgInfo(dtype=torch.float16, grad_of="output"),
                grad_weight=ArgInfo(
                    dtype=torch.float16, init=Init.ZERO, grad_of="weight"
                ),
            ),
            args=["grad_weight", "input", "grad_output"],
            kernel_outputs=["grad_weight"],
            reference=torch.nn.functional.conv2d,
            prefix_op_constructor=_make_prefix_op,
            stats=Conv2dStats,
        )
    )

    register_reflection(
        Reflection(
            kernel_cls=Conv2dGw8Kernel,
            kwargs=dict(igrad=True),
            arginfo=dict(
                input=ArgInfo(dtype=torch.float16, requires_grad=True),
                weight=ArgInfo(dtype=torch.float16, requires_grad=True),
                bias=ArgInfo(dtype=torch.float16, requires_grad=True),
                output=ArgInfo(dtype=torch.float16, output=True, init=Init.EMPTY),
                grad_output=ArgInfo(dtype=torch.float16, grad_of="output"),
                grad_input=ArgInfo(
                    dtype=torch.float16, init=Init.EMPTY, grad_of="input"
                ),
                none=ArgInfo(dtype=torch.float16, init=Init.NONE),
            ),
            args=["grad_input", "grad_output", "weight", "none"],
            kernel_outputs=["grad_input"],
            reference=torch.nn.functional.conv2d,
            stats=Conv2dStats,
            stacking=[("grad_input", "grad_output")],
            prefix_op_constructor=_make_prefix_op,
        )
    )


def _make_prefix_op(params: Conv2dGw8Params) -> torch.nn.Module:
    """Return a prefix operation for the kernel.

    Prefix operations are used to cover the latency of launching the benchmark kernels.
    This improves the accuracy of the benchmark when the kernels have low latency.
    """
    TARGET_LATENCY = (
        1e-3  # 1e-3 seconds = 1 ms (plenty of time to hide kernel launch overhead).
    )
    PEAK_ARITHMETIC_THROUGHPUT = 150e12  # 150 TFLOPS (realistic upper bound for GPUs)
    TARGET_MACS = TARGET_LATENCY * PEAK_ARITHMETIC_THROUGHPUT

    N, C, H, W = params.N, params.C, params.H, params.W
    macs_per_kernel_dim = (N * C * H * W) * C
    kernel_size = _next_odd_number(math.sqrt(TARGET_MACS / macs_per_kernel_dim))
    padding = kernel_size // 2
    return torch.nn.Conv2d(C, C, kernel_size=kernel_size, padding=padding).cuda().half()


def _next_odd_number(x: float) -> int:
    """Return the next odd number greater than or equal to x."""
    return math.ceil(x) // 2 * 2 + 1


def _torch_conv2d_kwargs(params):
    """Return the keyword arguments for torch.nn.function.conv2d"""
    return dict(stride=params.stride, padding=params.padding, groups=params.groups)


def _spio_conv2d_gw8_kwargs(params):
    """Return the keyword arguments for spio.functional.conv2d_gw8"""
    return dict(
        stride=params.stride,
        padding_y=params.padding_h,
        padding_x=params.padding_w,
        groups=params.groups,
    )

def _conv2d_layer_kwargs(params):
    """Return the keyword arguments for spio.layers.Conv2dGw8 and torch.nn.Conv2d"""
    return dict(
        in_channels=params.C,
        out_channels=params.K,
        kernel_size=params.kernel_size,
        stride=params.stride,
        padding=params.padding,
        groups=params.groups,
        bias=params.has_bias,
    )