"""A module for the MlpParams class."""

from dataclasses import dataclass
import re
from typing import Tuple
from math import prod

import torch


@dataclass(frozen=True)
class MlpParams:
    """Parameters for a multi-layer perceptron (MLP) operation.

    This class is used to validate and store parameters for an MLP operation.

    Attributes:
        x (int): Input samples.
        c (int): Input channels.
        r (int): Hidden channels.
        k (int): Output channels.
        bias (bool): Whether to apply a bias term (default is True).
        activation (str): Activation function to use.
    """

    x: int
    c: int
    r: int
    k: int
    bias: bool = True
    activation: str = "relu"

    def encode(self) -> str:
        """Return a string representation of the parameters."""
        bias_str = "b" if self.bias else "nb"
        return f"{self.x}x_{self.c}c_{self.r}r_{self.k}k_{bias_str}_{self.activation}"

    @classmethod
    def decode(cls, string: str) -> "MlpParams":
        """Get a MlpParams instance from a string rep."""
        matches = re.match(r"(\d+)x_(\d+)c_(\d+)r_(\d+)k_(b|nb)_(.*)", string)
        if matches is None:
            raise ValueError("Invalid string representation.")
        return MlpParams(
            x=int(matches.group(1)),
            c=int(matches.group(2)),
            r=int(matches.group(3)),
            k=int(matches.group(4)),
            bias=(matches.group(5) == "b"),
            activation=matches.group(6),
        )

    @staticmethod
    def from_tensors(
        inputs: torch.Tensor,
        exp_weight: torch.Tensor,
        exp_bias: torch.Tensor,
        prj_weight: torch.Tensor,
        prj_bias: torch.Tensor,
        activation: str = "relu",
    ):
        """Create a MlpParams instance from tensors."""
        *samples, c_in = inputs.shape
        x = prod(samples)
        r_exp, c_exp = exp_weight.shape
        k_prj, r_prj = prj_weight.shape
        bias = exp_bias is not None
        if bias:
            assert prj_bias is not None
            assert exp_bias.shape == (r_exp,)
            assert prj_bias.shape == (k_prj,)
        assert c_exp == c_in
        assert r_prj == r_exp
        return MlpParams(
            x=x, c=c_in, r=r_exp, k=k_prj, bias=bias, activation=activation
        )

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Return the shape of the input tensor."""
        return self.x, self.c

    @property
    def output_shape(self) -> Tuple[int, int]:
        """Return the shape of the output tensor."""
        return self.x, self.k

    @property
    def exp_weight_shape(self) -> Tuple[int, int]:
        """Return the shape of the expansion weight tensor."""
        return self.r, self.c

    @property
    def exp_bias_shape(self) -> Tuple[int]:
        """Return the shape of the expansion bias tensor."""
        return (self.r,)

    @property
    def prj_weight_shape(self) -> Tuple[int, int]:
        """Return the shape of the projection weight tensor."""
        return self.k, self.r

    @property
    def prj_bias_shape(self) -> Tuple[int]:
        """Return the shape of the projection bias tensor."""
        return (self.k,)
