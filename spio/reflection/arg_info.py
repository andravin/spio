"""Argument information for kernels and functions."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from ..kernels import Params


class Init(Enum):
    """Initialization types for arguments."""

    ZERO = 0
    ONE = 1
    NONE = 2
    EMPTY = 3
    RANDOM = 4


@dataclass
class ArgInfo:
    """Information about an argument to a kernel or function."""

    dtype: torch.dtype
    requires_grad: bool = False
    output: bool = False
    init: Init = Init.RANDOM
    memory_format: torch.memory_format = torch.channels_last
    grad_of: str = None
    name: str = None
    scale: float = 1.0

    @property
    def zero(self) -> bool:
        """Return true if the initialization type is zero."""
        return self.init == Init.ZERO

    @property
    def one(self) -> bool:
        """Return true if the initialization type is one."""
        return self.init == Init.ONE

    @property
    def none(self) -> bool:
        """Return true if the initialization type is none."""
        return self.init == Init.NONE

    @property
    def empty(self) -> bool:
        """Return true if the initialization type is empty."""
        return self.init == Init.EMPTY

    @property
    def random(self) -> bool:
        """Return true if the initialization type is random."""
        return self.init == Init.RANDOM

    def _shape(self, params: Params):
        if self.grad_of is not None:
            name = self.grad_of
        else:
            name = self.name
        return getattr(params, f"{name}_shape")

    def _empty(self, params: Params, device: str) -> torch.Tensor:
        return torch.empty(
            self._shape(params),
            dtype=self.dtype,
            device=device,
            memory_format=self.memory_format,
        )

    def _zero(self, params: Any, device: str) -> torch.Tensor:
        t = torch.zeros(self._shape(params), dtype=self.dtype, device=device)
        return _to(t, memory_format=self.memory_format)

    def _ones(self, params: Any, device: str) -> torch.Tensor:
        t = (
            torch.ones(self._shape(params), dtype=self.dtype, device=device)
            * self.scale
        )
        return _to(t, memory_format=self.memory_format)

    def _randn_clip_3(self, params: Any, device: str) -> torch.Tensor:
        shape = self._shape(params)
        t = torch.randn(shape, dtype=self.dtype, device=device)
        t = torch.clip(t, -3, 3) * self.scale
        return _to(t, memory_format=self.memory_format)

    def make_arg(self, params: Params, training=False, device="cuda") -> torch.Tensor:
        """Make an argument tensor based on the ArgInfo settings."""
        has_arg = _has_arg(params, self.name)
        if not has_arg or self.none:
            tensor = None
        elif self.empty:
            tensor = self._empty(params, device)
        elif self.zero:
            tensor = self._zero(params, device)
        elif self.one:
            tensor = self._ones(params, device)
        elif self.random:
            tensor = self._randn_clip_3(params, device)
        else:
            raise ValueError(
                f"Invalid init value for argument {self.name}: {self.init}"
            )
        if self.requires_grad and training and has_arg:
            tensor.requires_grad = True
        return tensor

    def initialize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Initialize the given tensor based on the ArgInfo settings."""
        if tensor is None:
            assert self.none
            return
        if self.zero:
            tensor.zero_()
        elif self.one:
            tensor.fill_(1)
        elif self.random:
            tensor.normal_()


def _has_arg(params, name) -> bool:
    return getattr(params, f"has_{name}", True)


def _to(tensor, memory_format=None) -> torch:
    """Normalize the memory format of a tensor.

    channels_last is only supported for 4D tensors. We leave 1D tensors
    unchanged.

    Otherwise, we raise an error if the tensor is not 4D.
    """
    if tensor.dim() < 2:
        # You didn't really mean it.
        return tensor

    if memory_format == torch.channels_last and tensor.dim() != 4:
        # You meant it, but it's not supported.
        raise ValueError("channels_last memory format is only supported for 4D tensors")

    return tensor.to(memory_format=memory_format)
