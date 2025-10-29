"""A module for the LayerNorm2DParams class."""

from dataclasses import dataclass
import re
from typing import Tuple

import torch


@dataclass(frozen=True)
class LayerNorm2dParams:
    """Parameters for a 2D layer normalization operation.

    This class is used to validate and store parameters for a 2D layer normalization operation.

    Attributes:
        n (int): Batch size.
        c (int): Number of channels.
        h (int): Height of the input.
        w (int): Width of the input.
        elementwise_affine (bool): Whether to learn a weight and bias for each channel.
        bias (bool): Whether to apply a bias term (default is True).
    """

    n: int
    c: int
    h: int
    w: int
    elementwise_affine: bool = True
    bias: bool = True
    eps: float = 1e-5

    @property
    def has_weight(self) -> bool:
        """Return true if the layer has a weight term.

        The layer has a weight term if the elementwise_affine attribute is True.
        """
        return self.elementwise_affine

    @property
    def has_bias(self) -> bool:
        """Return true if the layer has a bias term.

        The layer has a bias term if the bias and elementwise_affine attributes are both True.
        """
        return self.elementwise_affine and self.bias

    def encode(self) -> str:
        """Return a string representation of the parameters."""
        bias_str = "b" if self.bias else "nb"
        elementwise_affine_str = "ea" if self.elementwise_affine else "nea"
        return f"{self.n}n_{self.c}c_{self.h}h_{self.w}_{bias_str}_{elementwise_affine_str}"

    @classmethod
    def decode(cls, string: str) -> "LayerNorm2dParams":
        """Get a LayerNorm2dParams instance from a string rep."""
        matches = re.match(r"(\d+)n_(\d+)c_(\d+)h_(\d+)w_(b|nb)_(ea|nea)", string)
        if matches is None:
            raise ValueError("Invalid string representation.")
        groups = matches.groups()
        n, c, h, w = map(int, groups[:4])
        bias = groups[4] == "b"
        elementwise_affine = groups[5] == "ea"
        return cls(n=n, c=c, h=h, w=w, bias=bias, elementwise_affine=elementwise_affine)

    @staticmethod
    def from_tensors(
        inputs: torch.Tensor,
        weight: torch.Tensor = None,
        bias: torch.Tensor = None,
        eps: float = 1e-5,
    ):
        """Derive a LayerNorm2dParams instance from tensor args."""
        n, c, h, w = inputs.shape
        return LayerNorm2dParams(
            n=n,
            c=c,
            h=h,
            w=w,
            elementwise_affine=weight is not None,
            bias=bias is not None,
            eps=eps,
        )

    def is_valid(self) -> bool:
        """Return true if the parameters are valid."""
        try:
            self.validate()
            return True
        except AssertionError:
            return False

    def validate(self):
        """Assert that the parameters are valid."""
        assert self.n > 0, "Batch size must be positive."
        assert self.c > 0, "Number of channels must be positive."
        assert self.h > 0, "Height must be positive."
        assert self.w > 0, "Width must be positive."
        assert self.bias in [True, False], "Bias must be a boolean."
        assert self.elementwise_affine in [
            True,
            False,
        ], "Elementwise affine must be a boolean."
        assert self.c <= 2048, "Number of channels must be less than 2048."
        assert self.c % 8 == 0, "Number of channels must be a multiple of 8."

    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        """Return the shape of the input tensor as a tuple."""
        return (self.n, self.c, self.h, self.w)

    @property
    def output_shape(self) -> Tuple[int, int, int, int]:
        """Return the shape of the output tensor as a tuple.

        The output shape is the same as the input shape.
        """
        return self.input_shape

    @property
    def weight_shape(self) -> Tuple[int]:
        """Return the shape of the weight tensor as a tuple."""
        return (self.c,) if self.elementwise_affine else None

    @property
    def bias_shape(self) -> Tuple[int]:
        """Return the shape of the bias tensor as a tuple."""
        return (self.c,) if self.has_bias else None
