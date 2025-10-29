"""Custom layernorm_2d operator for PyTorch."""

import torch
import torch.amp

from ..kernels import layernorm_2d_kernel_factory, LayerNorm2dParams
from ..util import to_channels_last


@torch.library.custom_op("spio::layernorm_2d", mutates_args=())
def layernorm_2d(
    inputs: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Custom layernorm_2d function.

    Implements layernorm_2d with float16 precision.
    """
    assert inputs.dtype == torch.float16
    assert weight is None or weight.dtype == torch.float32
    assert bias is None or bias.dtype == torch.float32
    params = LayerNorm2dParams.from_tensors(inputs, weight=weight, bias=bias, eps=eps)
    output = torch.empty_like(inputs, memory_format=torch.channels_last)
    args = (output, inputs, weight, bias)
    args = to_channels_last(*args)
    kernel = layernorm_2d_kernel_factory.get_kernel(params, inputs.device)
    kernel(*args)
    return output


def layernorm_2d_autocast(ks, inputs, weight, bias, *args, **kwargs):
    """Cast the layernorm_2d arguments list."""
    input_dtype = inputs.dtype
    inputs = inputs.to(dtype=torch.float16)
    if weight is not None:
        weight = weight.to(dtype=torch.float32)
    if bias is not None:
        bias = bias.to(dtype=torch.float32)
    autocast = torch._C.DispatchKeySet("AutocastCUDA")
    with torch._C._ExcludeDispatchKeyGuard(autocast):
        result = torch.ops.spio.layernorm_2d.default.redispatch(
            ks - autocast, inputs, weight, bias, *args, **kwargs
        )
    result = result.to(dtype=input_dtype)
    return result


# pylint: disable=unused-argument
@layernorm_2d.register_fake
def _(inputs, weight=None, bias=None, eps=1e-5):
    """FakeTensor implementation of layernorm_2d."""
    return inputs.new_empty(inputs.shape).to(memory_format=torch.channels_last)


# See discussion at https://github.com/pytorch/pytorch/issues/137033
m = torch.library.Library("spio", "FRAGMENT")
m.impl("layernorm_2d", layernorm_2d_autocast, "AutocastCUDA", with_keyset=True)
