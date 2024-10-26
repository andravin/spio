"""Make spio modules."""

from .conv2d_gw8 import Conv2dGw8


def make_conv2d(*args, **kwargs):
    """Create a spio module for the given torch.nn.Conv2d arguments.

    Returns a spio module that implements the torch.nn.Conv2d
    functionality.

    If the arguments do not satisfy the requirements for any of the
    available spio modules, returns None.
    """
    return Conv2dGw8.make(*args, **kwargs)
