"""Make spio modules."""

from .conv2d_gw8 import Conv2dGw8
from .layernorm_2d import LayerNorm2d


def make_conv2d(*args, **kwargs):
    """Create a spio module for the given torch.nn.Conv2d arguments.

    Returns a spio module that implements the torch.nn.Conv2d
    functionality.

    If the arguments do not satisfy the requirements for any of the
    available spio modules, returns None.
    """
    return Conv2dGw8.make(*args, **kwargs)


def make_layernorm_2d(*args, **kwargs):
    """Create a spio LayerNorm2d module for the given torch.nn.LayerNorm arguments.

    Returns a spio module that implements a LayerNorm2d layer.

    If the arguments do not satisfy the requirements for the LayerNorm2d moduloe,
    returns None.
    """
    return LayerNorm2d.make(*args, **kwargs)
