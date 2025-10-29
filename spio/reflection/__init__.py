"""Registry of kernel, function, and layer reflections."""

from .reflection import (
    get_kernel_reflection,
    get_function_reflection,
    get_layer_reflection,
)
from .conv2d_gw8_reflection import register_conv2d_gw8_reflections
from .layernorm_2d_reflection import register_layernorm_2d_reflections

register_conv2d_gw8_reflections()
register_layernorm_2d_reflections()