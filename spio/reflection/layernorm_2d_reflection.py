"""Reflections for layernorm_2d kernels and functions."""

import torch

from ..kernels.layernorm_2d_stats import LayerNorm2dStats
from ..kernels.layernorm_2d_params import LayerNorm2dParams
from ..functional.layernorm_2d_function import layernorm_2d

from .reflection import Reflection, ArgInfo, register_reflection, Init


_layernorm_2d_arginfo = {
    "input": ArgInfo(dtype=torch.float16, requires_grad=True),
    "output": ArgInfo(dtype=torch.float16, output=True, init=Init.EMPTY),
    "weight": ArgInfo(dtype=torch.float32, requires_grad=True),
    "bias": ArgInfo(dtype=torch.float32, requires_grad=True),
}


def _layer_norm_2d_reference(
    inputs: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps=1e-5,
):
    c = inputs.shape[1]
    normalized_shape = (c,)
    input_nhwc = inputs.permute(0, 2, 3, 1)
    output_nhwc = torch.nn.functional.layer_norm(
        input_nhwc, normalized_shape, weight=weight, bias=bias, eps=eps
    )
    output_nchw = output_nhwc.permute(0, 3, 1, 2)
    return output_nchw


def _layer_norm_2d_reference_kwargs(_params):
    return {}


def register_layernorm_2d_reflections():
    """Register reflections for layernorm_2d."""

    register_reflection(
        Reflection(
            kernel_name="spio_layernorm_2d",
            arginfo=_layernorm_2d_arginfo,
            args=["output", "input", "weight", "bias"],
            kernel_outputs=["output"],
            reference=_layer_norm_2d_reference,
            stats=LayerNorm2dStats,
            params=LayerNorm2dParams,
        )
    )

    register_reflection(
        Reflection(
            function=layernorm_2d,
            arginfo=_layernorm_2d_arginfo,
            args=["input", "weight", "bias"],
            reference=_layer_norm_2d_reference,
            stats=LayerNorm2dStats,
            params=LayerNorm2dParams,
            get_function_kwargs=_spio_layernorm_2d_kwargs,
        )
    )

    register_reflection(
        Reflection(
            function=_layer_norm_2d_reference,
            arginfo=_layernorm_2d_arginfo,
            args=["input", "weight", "bias"],
            stats=LayerNorm2dStats,
            params=LayerNorm2dParams,
            get_function_kwargs=_layer_norm_2d_reference_kwargs,
        )
    )


def _spio_layernorm_2d_kwargs(params):
    return { "eps": params.eps }
