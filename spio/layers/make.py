from .conv2d_gw8 import Conv2dGw8

def make_conv2d(*args, **kwargs):
    """Create a spio module for the given torch.nn.Conv2d arguments if possible.
    
    Returns None if the arguments do not match the requirements. Otherwise, returns a spio module
    that implements the torch.nn.Conv2d functionality.
    """
    return Conv2dGw8.make(*args, **kwargs)
