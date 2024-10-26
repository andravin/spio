"""Registry of kernel, function, and layer reflections."""

from .reflection import (
    get_kernel_reflection,
    get_function_reflection,
    get_layer_reflection,
)
from .conv2d_gw8_reflection import register_conv2d_gw8_reflections

register_conv2d_gw8_reflections()
