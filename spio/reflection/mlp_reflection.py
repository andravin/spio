"""Reflection for mlp kernels and functions."""

import torch
import torch.nn.functional as F

from ..kernels.mlp_stats import MlpStats
from ..kernels.mlp_params import MlpParams
from ..util import SixteenChannelsLast

from .reflection import Reflection, ArgInfo, register_reflection, Init


def _mlp_reference(
    inputs: torch.Tensor,
    exp_weight: torch.Tensor,
    exp_bias: torch.Tensor = None,
    prj_weight: torch.Tensor = None,
    prj_bias: torch.Tensor = None,
    activation: torch.nn.Module = None,
):
    # pylint: disable=not-callable
    # pylint does not see the pytorch .pyi declaration for F.linear:
    # https://github.com/pytorch/pytorch/issues/119482
    hidden = F.linear(inputs, exp_weight, bias=exp_bias)
    if activation is not None:
        hidden = activation(hidden)
    output = F.linear(hidden, prj_weight, bias=prj_bias)
    return output


def _mlp_reference_kwargs(params: MlpParams):
    """Return kwargs for the reference function."""
    return {
        "activation": _reference_activation(params.activation),
    }


def register_mlp_reflections():
    """Register reflections for mlp."""

    mlp_kernel_arg_info = {
        "output": ArgInfo(dtype=torch.float16, output=True, init=Init.EMPTY),
        "input": ArgInfo(dtype=torch.float16, requires_grad=True),
        "exp_weight": ArgInfo(
            dtype=torch.float16, requires_grad=True, memory_format=SixteenChannelsLast
        ),
        "exp_bias": ArgInfo(dtype=torch.float32, requires_grad=True),
        "prj_weight": ArgInfo(
            dtype=torch.float16, requires_grad=True, memory_format=SixteenChannelsLast
        ),
        "prj_bias": ArgInfo(dtype=torch.float32, requires_grad=True),
    }

    mlp_reference_arg_info = {
        "output": ArgInfo(dtype=torch.float16, output=True, init=Init.EMPTY),
        "input": ArgInfo(dtype=torch.float16, requires_grad=True),
        "exp_weight": ArgInfo(
            dtype=torch.float16,
            requires_grad=True,
        ),
        "exp_bias": ArgInfo(dtype=torch.float32, requires_grad=True),
        "prj_weight": ArgInfo(
            dtype=torch.float16,
            requires_grad=True,
        ),
        "prj_bias": ArgInfo(dtype=torch.float32, requires_grad=True),
    }

    register_reflection(
        Reflection(
            kernel_name="mlp_tiny_c",
            arginfo=mlp_kernel_arg_info,
            args=[
                "output",
                "input",
                "exp_weight",
                "exp_bias",
                "prj_weight",
                "prj_bias",
            ],
            kernel_outputs=["output"],
            reference=_mlp_reference,
            stats=MlpStats,
            params=MlpParams,
        )
    )

    register_reflection(
        Reflection(
            function=_mlp_reference,
            arginfo=mlp_reference_arg_info,
            args=[
                "input",
                "exp_weight",
                "exp_bias",
                "prj_weight",
                "prj_bias",
            ],
            stats=MlpStats,
            params=MlpParams,
            get_function_kwargs=_mlp_reference_kwargs,
        )
    )


def _reference_activation(activation: str):
    if activation == "relu":
        return torch.nn.ReLU()
    if activation == "gelu":
        return torch.nn.GELU()
    if activation == "silu":
        return torch.nn.SiLU()
    raise ValueError(f"Unsupported activation function: {activation}")
