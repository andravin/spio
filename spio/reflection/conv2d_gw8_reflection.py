"""Reflections for conv2d_gw8 kernels and functions."""

import torch

from ..kernels.conv2d_stats import Conv2dStats
from ..kernels.conv2d_gw8_params import Conv2dGw8Params
from ..functional.conv2d_gw8_function import conv2d_gw8
from ..layers.conv2d_gw8 import Conv2dGw8

from .reflection import Reflection, ArgInfo, register_reflection, Init


def register_conv2d_gw8_reflections():
    """Register reflections for conv2d_gw8."""

    # ---------------------------------------------------------------------------------------------

    conv2d_arg_info = {
        "output": ArgInfo(dtype=torch.float16, output=True, init=Init.EMPTY),
        "input": ArgInfo(dtype=torch.float16, requires_grad=True),
        "weight": ArgInfo(dtype=torch.float16, requires_grad=True),
        "bias": ArgInfo(dtype=torch.float32, requires_grad=True),
    }

    register_reflection(
        Reflection(
            kernel_name="spio_conv2d_gw8_fprop",
            arginfo=conv2d_arg_info,
            args=["output", "input", "weight", "bias"],
            kernel_outputs=["output"],
            reference=torch.nn.functional.conv2d,
            stats=Conv2dStats,
            params=Conv2dGw8Params,
            ignore_params=["stride", "group_width"],
        )
    )

    register_reflection(
        Reflection(
            function=conv2d_gw8,
            arginfo=conv2d_arg_info,
            args=["input", "weight", "bias"],
            reference=torch.nn.functional.conv2d,
            stats=Conv2dStats,
            params=Conv2dGw8Params,
            get_function_kwargs=_spio_conv2d_gw8_kwargs,
        )
    )

    register_reflection(
        Reflection(
            function=torch.nn.functional.conv2d,
            arginfo=conv2d_arg_info,
            args=["input", "weight", "bias"],
            reference=None,
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
            get_function_kwargs=_conv2d_layer_kwargs,
        )
    )

    # ---------------------------------------------------------------------------------------------

    register_reflection(
        Reflection(
            kernel_name="spio_conv2d_gw8_wgrad",
            arginfo={
                "input": ArgInfo(dtype=torch.float16, requires_grad=True),
                "weight": ArgInfo(dtype=torch.float16, requires_grad=True),
                "bias": ArgInfo(dtype=torch.float32, requires_grad=True),
                "output": ArgInfo(dtype=torch.float16, output=True, init=Init.EMPTY),
                "grad_output": ArgInfo(dtype=torch.float16, grad_of="output"),
                "grad_weight": ArgInfo(
                    dtype=torch.float32, init=Init.ZERO, grad_of="weight"
                ),
            },
            args=["grad_weight", "input", "grad_output"],
            kernel_outputs=["grad_weight"],
            reference=torch.nn.functional.conv2d,
            stats=Conv2dStats,
            params=Conv2dGw8Params,
            ignore_params=["stride", "group_width"],
        )
    )

    register_reflection(
        Reflection(
            kernel_name="spio_conv2d_gw8_dgrad",
            kwargs={"igrad": True},
            arginfo={
                "input": ArgInfo(dtype=torch.float16, requires_grad=True),
                "weight": ArgInfo(dtype=torch.float16, requires_grad=True),
                "bias": ArgInfo(dtype=torch.float32, requires_grad=True),
                "output": ArgInfo(dtype=torch.float16, output=True, init=Init.EMPTY),
                "grad_output": ArgInfo(dtype=torch.float16, grad_of="output"),
                "grad_input": ArgInfo(
                    dtype=torch.float16, init=Init.EMPTY, grad_of="input"
                ),
                "none": ArgInfo(dtype=torch.float16, init=Init.NONE),
            },
            args=["grad_input", "grad_output", "weight", "none"],
            kernel_outputs=["grad_input"],
            reference=torch.nn.functional.conv2d,
            stats=Conv2dStats,
            params=Conv2dGw8Params,
            ignore_params=["stride", "group_width"],
        )
    )


def _torch_conv2d_kwargs(params):
    """Return the keyword arguments for torch.nn.function.conv2d."""
    return {"stride": params.stride, "padding": params.padding, "groups": params.groups}


def _spio_conv2d_gw8_kwargs(params):
    """Return the keyword arguments for spio.functional.conv2d_gw8."""
    return {
        "stride": params.stride,
        "padding_y": params.padding_h,
        "padding_x": params.padding_w,
        "groups": params.groups,
    }


def _conv2d_layer_kwargs(params):
    """The keyword arguments for spio.layers.Conv2dGw8.

    These arguments also work with torch.nn.Conv2d.
    """
    return {
        "in_channels": params.c,
        "out_channels": params.k,
        "kernel_size": params.kernel_size,
        "stride": params.stride,
        "padding": params.padding,
        "groups": params.groups,
        "bias": params.has_bias,
    }
