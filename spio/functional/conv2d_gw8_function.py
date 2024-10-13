from typing import List, Tuple, Optional
import functools

import torch
import torch.amp

from ..kernels import Conv2dGw8Kernel, Conv2dGw8WgradKernel, Conv2dGw8Params


def _custom_setup_context(
    setup_context_fn=None,
    *,
    device_type: str,
    cast_inputs: Optional[torch.dtype] = None,
):
    """The missing amp setup_context decorator for custom ops.

    See https://github.com/pytorch/pytorch/issues/132388 for more details.
    """
    if setup_context_fn is None:
        return functools.partial(
            _custom_setup_context, device_type=device_type, cast_inputs=cast_inputs
        )

    @functools.wraps(setup_context_fn)
    def decorate_setup_context(ctx, *args, **kwargs):
        ctx._dtype = torch.get_autocast_dtype(device_type)
        if cast_inputs is None:
            ctx._fwd_used_autocast = torch.is_autocast_enabled(device_type)
            return setup_context_fn(ctx, *args, **kwargs)
        else:
            autocast_context = torch.is_autocast_enabled(device_type)
            ctx._fwd_used_autocast = False
            if autocast_context:
                with torch.autocast(device_type=device_type, enabled=False):
                    return setup_context_fn(
                        ctx,
                        *torch.amp.autocast_mode._cast(args, device_type, cast_inputs),
                        **torch.amp.autocast_mode._cast(kwargs, device_type, cast_inputs),
                    )
            else:
                return setup_context_fn(ctx, *args, **kwargs)

    return decorate_setup_context


@torch.library.custom_op("spio::conv2d_gw8", mutates_args=())
@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def conv2d_gw8(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding_y: int = 0,
    padding_x: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    assert input.dtype == torch.float16
    assert weight.dtype == torch.float16
    assert bias is None or bias.dtype == torch.float16
    params = Conv2dGw8Params.from_tensors(input, weight, bias, (padding_y, padding_x))
    output = torch.empty(
        params.output_shape,
        device=input.device,
        dtype=torch.float16,
        memory_format=torch.channels_last,
    )
    if bias is None:
        bias = _none(input.device)
    args = (output, input, weight, bias)
    kernel = Conv2dGw8Kernel.fprop_kernel(params, input.device)
    kernel(*args)
    return output


@conv2d_gw8.register_fake
def _(
    input,
    weight,
    bias=None,
    stride: int = 1,
    padding_y: int = 0,
    padding_x: int = 0,
    dilation=1,
    groups=1,
):
    params = Conv2dGw8Params.from_tensors(input, weight, bias, (padding_y, padding_x))
    return input.new_empty(params.output_shape).to(memory_format=torch.channels_last)


@torch.library.custom_op("spio::conv2d_gw8_backward", mutates_args=())
def conv2d_gw8_backward_op(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
    padding_y: int,
    padding_x: int,
    needs_input_grad: bool,
    needs_weight_grad: bool,
    needs_bias_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert grad_output.dtype == torch.float16
    assert input.dtype == torch.float16
    assert weight.dtype == torch.float16

    params = Conv2dGw8Params.from_tensors(
        input, weight, bias, padding=(padding_y, padding_x)
    )

    if needs_weight_grad:
        # The grad_weight kernel requires that the grad_weight tensor is initialized to zero.
        # Its data-type is torch.float32.
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        args = (grad_weight, input, grad_output)
        grad_weight_kernel = Conv2dGw8WgradKernel.grad_weight_kernel(
            params, input.device
        )
        grad_weight_kernel(*args)
    else:
        grad_weight = None

    if needs_bias_grad:
        # Grad_bias also uses torch.float32.
        # TODO: Incorporate grad_bias into the grad_weight_kernel.
        grad_bias = grad_output.sum(dim=(0, 2, 3), dtype=torch.float32)
    else:
        grad_bias = None

    if needs_input_grad:
        grad_input = torch.empty_like(input)
        args = (grad_input, grad_output, weight, _none(input.device))
        grad_input_kernel = Conv2dGw8Kernel.grad_input_kernel(params, input.device)
        grad_input_kernel(*args)
    else:
        grad_input = None
    return grad_input, grad_weight, grad_bias


@conv2d_gw8_backward_op.register_fake
def _(
    input,
    weight,
    bias,
    grad_output,
    padding_y,
    padding_x,
    needs_input_grad,
    needs_weight_grad,
    needs_bias_grad,
):
    params = Conv2dGw8Params.from_tensors(
        input, weight, bias, padding=(padding_y, padding_x)
    )
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
        grad_input = input.new_empty(input.shape).to(memory_format=torch.channels_last)
    else:
        grad_input = None
    return grad_input, grad_weight, grad_bias


@torch.amp.custom_bwd(device_type="cuda")
def conv2d_gw8_backward(ctx, grad_output):
    input, weight, bias = ctx.saved_tensors

    padding_y = ctx.padding_y
    padding_x = ctx.padding_x

    needs_input_grad = ctx.needs_input_grad[0]
    needs_weight_grad = ctx.needs_input_grad[1]
    needs_bias_grad = ctx.needs_input_grad[2]

    grad_input, grad_weight, grad_bias = conv2d_gw8_backward_op(
        input,
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


@_custom_setup_context(device_type="cuda", cast_inputs=torch.float16)
def conv2d_gw8_setup_context(ctx, inputs, output):
    """Setup the context for the conv2d_gw8 custom op.

    Note: Our _custom_setup_context decorator is a workaround for the missing amp setup_context decorator for custom ops.
    It has limited support for torch.compile. It does not cast the input and weight tensors are cast to float16.
    We use an assert to ensure that the input and weight tensors are in float16.
    """
    input, weight, bias, stride, padding_y, padding_x, *_ = inputs

    # Ensure that the weight tensor is in float16.
    assert input.dtype == torch.float16
    assert weight.dtype == torch.float16
    if bias is not None:
        assert bias.dtype == torch.float16

    ctx.save_for_backward(input, weight, bias)
    ctx.padding_y = padding_y
    ctx.padding_x = padding_x


conv2d_gw8.register_autograd(
    conv2d_gw8_backward, setup_context=conv2d_gw8_setup_context
)


def _none(device):
    """Return an empty tensor.

    This might not be necesary now that our spio.cuda.driver.Function.launch supports None arguments.
    """
    return torch.tensor([], device=device, dtype=torch.float16, requires_grad=False)
