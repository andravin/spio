"""Function API for spio operators.

These spio functions are PyTorch custom operators that work with autograd, amp, and
torch.compile. They uses the PyTorch Custom Operator API:
https://pytorch.org/tutorials/advanced/python_custom_ops.html
"""

from .conv2d_gw8_function import conv2d_gw8
