"""Custom conv2d_gw8 operator for PyTorch."""

from typing import Tuple

import torch
import torch.amp

from ..kernels import (
    conv2d_gw8_kernel_factory,
    conv2d_gw8_wgrad_kernel_factory,
    Conv2dGw8Params,
)

from ..util import to_channels_last


@torch.library.custom_op("spio::conv2d_gw8", mutates_args=())
def conv2d_gw8(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding_y: int = 0,
    padding_x: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Custom conv2d_gw8 function.

    Implements conv2d with group width equal to 8. Uses channels last memory format
    and float16 precision.

    Args:
        inputs: Input tensor of shape (N, C, H, W).
        weight: Weight tensor of shape (C, 8, R, S).
        bias: Optional bias tensor of shape (C,).
        stride: Stride for the convolution. Must equal 1.
        padding_y: Padding for the height dimension.
        padding_x: Padding for the width dimension.
        dilation: Dilation for the convolution. Must equal 1.
        groups: Number of groups for the convolution. Must equal C // 8.

    Returns:
        Output tensor of shape (N, C, P, Q).
    """
    assert inputs.dtype == torch.float16
    assert weight.dtype == torch.float16
    assert bias is None or bias.dtype == torch.float32
    assert stride == 1 or stride == (1, 1)
    assert dilation == 1 or dilation == (1, 1)
    assert groups == inputs.shape[1] // 8
    params = Conv2dGw8Params.from_tensors(inputs, weight, bias, (padding_y, padding_x))
    output = torch.empty(
        params.output_shape,
        device=inputs.device,
        dtype=torch.float16,
        memory_format=torch.channels_last,
    )
    args = (output, inputs, weight, bias)
    args = to_channels_last(*args)
    kernel = conv2d_gw8_kernel_factory.get_kernel(params, inputs.device)
    kernel(*args)
    return output


# See discussion at https://github.com/pytorch/pytorch/issues/137033
def conv2d_gw8_autocast(ks, inputs, weight, bias, *args, **kwargs):
    """Autocast wrapper for conv2d_gw8."""
    input_dtype = inputs.dtype
    inputs = inputs.to(dtype=torch.float16)
    weight = weight.to(dtype=torch.float16)
    if bias is not None:
        bias = bias.to(dtype=torch.float32)
    autocast = torch._C.DispatchKeySet("AutocastCUDA")
    with torch._C._ExcludeDispatchKeyGuard(autocast):
        result = torch.ops.spio.conv2d_gw8.default.redispatch(
            ks - autocast, inputs, weight, bias, *args, **kwargs
        )
    result = result.to(dtype=input_dtype)
    return result


@conv2d_gw8.register_fake
def _(
    inputs,
    weight,
    bias=None,
    stride: int = 1,
    padding_y: int = 0,
    padding_x: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    """FakeTensor implementation of conv2d_gw8."""
    assert stride == 1 or stride == (1, 1)
    assert dilation == 1 or dilation == (1, 1)
    assert groups == inputs.shape[1] // 8
    params = Conv2dGw8Params.from_tensors(inputs, weight, bias, (padding_y, padding_x))
    return inputs.new_empty(params.output_shape).to(memory_format=torch.channels_last)


@torch.library.custom_op("spio::conv2d_gw8_backward", mutates_args=())
def conv2d_gw8_backward_op(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
    padding_y: int,
    padding_x: int,
    needs_input_grad: bool,
    needs_weight_grad: bool,
    needs_bias_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward function for conv2d_gw8."""

    assert grad_output.dtype == torch.float16
    assert inputs.dtype == torch.float16
    assert weight.dtype == torch.float16

    grad_output, inputs, weight = to_channels_last(grad_output, inputs, weight)

    params = Conv2dGw8Params.from_tensors(
        inputs, weight, bias, padding=(padding_y, padding_x)
    )

    if needs_weight_grad:
        # The grad_weight kernel requires that the grad_weight tensor is initialized to zero.
        # Its data-type is torch.float32.
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        args = (grad_weight, inputs, grad_output)
        grad_weight_kernel = conv2d_gw8_wgrad_kernel_factory.get_kernel(
            params, inputs.device
        )
        grad_weight_kernel(*args)
    else:
        grad_weight = None

    if needs_bias_grad:
        # Grad_bias also uses torch.float32.
        grad_bias = grad_output.sum(dim=(0, 2, 3), dtype=torch.float32)
    else:
        grad_bias = None

    if needs_input_grad:
        grad_input = torch.empty_like(inputs)
        args = (grad_input, grad_output, weight, None)
        grad_input_kernel = conv2d_gw8_kernel_factory.get_kernel(
            params, inputs.device, igrad=True
        )
        grad_input_kernel(*args)
    else:
        grad_input = None
    return grad_input, grad_weight, grad_bias


@conv2d_gw8_backward_op.register_fake
def _(
    inputs,
    weight,
    bias,
    _grad_output,
    _padding_y,
    _padding_x,
    needs_input_grad,
    needs_weight_grad,
    needs_bias_grad,
):
    """FakeTensor implementation of conv2d_gw8_backward."""
    if needs_weight_grad:
        grad_weight = weight.new_empty(weight.shape).to(
            memory_format=torch.channels_last
        )
    else:
        grad_weight = None

    if needs_bias_grad:
        grad_bias = bias.new_empty(bias.shape)
    else:
        grad_bias = None

    if needs_input_grad:
        grad_input = inputs.new_empty(inputs.shape).to(
            memory_format=torch.channels_last
        )
    else:
        grad_input = None
    return grad_input, grad_weight, grad_bias


def conv2d_gw8_backward(ctx, grad_output):
    """Backward function for conv2d_gw8."""
    inputs, weight, bias = ctx.saved_tensors

    padding_y = ctx.padding_y
    padding_x = ctx.padding_x

    needs_input_grad = ctx.needs_input_grad[0]
    needs_weight_grad = ctx.needs_input_grad[1]
    needs_bias_grad = ctx.needs_input_grad[2]

    grad_input, grad_weight, grad_bias = conv2d_gw8_backward_op(
        inputs,
        weight,
        bias,
        grad_output,
        padding_y,
        padding_x,
        needs_input_grad,
        needs_weight_grad,
        needs_bias_grad,
    )

    return grad_input, grad_weight, grad_bias, None, None, None, None, None


def conv2d_gw8_setup_context(ctx, inputs, output):
    """Setup the context for the conv2d_gw8 custom op."""
    input_tensor, weight, bias, _, padding_y, padding_x, *_ = inputs

    # Ensure that the tensor are float16 ..
    assert input_tensor.dtype == torch.float16
    assert weight.dtype == torch.float16
    assert output.dtype == torch.float16
    if bias is not None:
        # .. except for the bias, which is float32.
        assert bias.dtype == torch.float32

    ctx.save_for_backward(input_tensor, weight, bias)
    ctx.padding_y = padding_y
    ctx.padding_x = padding_x


conv2d_gw8.register_autograd(
    conv2d_gw8_backward, setup_context=conv2d_gw8_setup_context
)

# See discussion at https://github.com/pytorch/pytorch/issues/137033
m = torch.library.Library("spio", "FRAGMENT")
m.impl("conv2d_gw8", conv2d_gw8_autocast, "AutocastCUDA", with_keyset=True)
