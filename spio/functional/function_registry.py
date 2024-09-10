from .conv_gw8_function import conv2d_gw8

FUNCTION_REGISTRY = dict(
    conv2d_gw8=conv2d_gw8
)


def get_function(name):
    return FUNCTION_REGISTRY[name]
