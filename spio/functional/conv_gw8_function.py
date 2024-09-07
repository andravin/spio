import torch
import torch.amp
from torch.cuda.amp import custom_fwd, custom_bwd

import cupy

from ..kernels import Conv2dGw8Kernel, Conv2dGw8WgradKernel, Conv2dGw8Params


class ConvGw8Function(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        kernel_config=None,
    ):
        params = Conv2dGw8Params.from_tensors(input, weight, bias, padding=padding)
        ctx.save_for_backward(input, weight, bias)
        ctx.params = params
        output = torch.empty(
            params.output_shape,
            device=input.device,
            dtype=torch.float16,
            memory_format=torch.channels_last,
        )
        if bias is None:
            bias = _none(input.device)
        args = (output, input, weight, bias)
        kernel = Conv2dGw8Kernel.fprop_kernel(params, args, config=kernel_config)
        kernel(*args)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        params = ctx.params

        assert grad_output.dtype == torch.float16
        assert input.dtype == torch.float16
        assert weight.dtype == torch.float16

        if ctx.needs_input_grad[0]:
            dgrad = torch.empty_like(input)
            args = (dgrad, grad_output, weight, _none(input.device))
            dgrad_kernel = Conv2dGw8Kernel.dgrad_kernel(params, args)
            dgrad_kernel(*args)

        if ctx.needs_input_grad[1]:
            wgrad = torch.zeros_like(weight)
            args = (wgrad, input, grad_output)
            wgrad_kernel = Conv2dGw8WgradKernel.wgrad_kernel(params, args)
            wgrad_kernel(*args)

        if bias is not None and ctx.needs_input_grad[2]:
            bgrad = grad_output.sum(dim=(0, 2, 3))
        else:
            bgrad = None

        return dgrad, wgrad, bgrad, None, None, None, None


def conv2d_gw8(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return ConvGw8Function.apply(input, weight, bias, stride, padding, dilation, groups)


def _none(device):
    """Return an empty tensor.

    CuPy does not support None arguments, so we use this function to pass an empty tensor instead.
    """
    return torch.tensor([], device=device, dtype=torch.float16, requires_grad=False)
