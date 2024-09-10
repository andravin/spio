from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


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

    @property
    def zero(self):
        return self.init == Init.ZERO

    @property
    def one(self):
        return self.init == Init.ONE

    @property
    def none(self):
        return self.init == Init.NONE

    @property
    def empty(self):
        return self.init == Init.EMPTY

    @property
    def random(self):
        return self.init == Init.RANDOM

    def _shape(self, params):
        if self.grad_of is not None:
            name = self.grad_of
        else:
            name = self.name
        return getattr(params, f"{name}_shape")

    def _none(self):
        return None

    def _empty(self, params: Any, device: str):
        return torch.empty(
            self._shape(params),
            dtype=self.dtype,
            device=device,
            memory_format=self.memory_format,
        )

    def _zero(self, params: Any, device: str):
        t = torch.zeros(self._shape(params), dtype=self.dtype, device=device)
        return _to(t, memory_format=self.memory_format)

    def _ones(self, params: Any, device: str):
        t = torch.ones(self._shape(params), dtype=self.dtype, device=device)
        return _to(t, memory_format=self.memory_format)

    def _randn(self, params: Any, device: str):
        shape = self._shape(params)
        t = torch.randn(shape, dtype=self.dtype, device=device)
        return _to(t, memory_format=self.memory_format)

    def make_arg(self, params, training=False, device="cuda"):
        has_arg = _has_arg(params, self.name)
        if not has_arg or self.none:
            tensor = self._none()
        elif self.empty:
            tensor = self._empty(params, device)
        elif self.zero:
            tensor = self._zero(params, device)
        elif self.one:
            tensor = self._ones(params, device)
        elif self.random:
            tensor = self._randn(params, device)
        else:
            raise ValueError(f"Invalid init value for argument {self.name}: {self.init}")
        if self.requires_grad and training and has_arg:
            tensor.requires_grad = True
        return tensor
    
    def initialize(self, tensor:torch.Tensor):
        if tensor is None:
            assert self.none
            return
        if self.zero:
            tensor.zero_()
        elif self.one:
            tensor.fill_(1)
        elif self.random:
            tensor.normal_()


def _has_arg(params, name):
    return getattr(params, f"has_{name}", True)


def _to(tensor, memory_format=None):
    """Normalize the memory format of a tensor.

    channels_last is only supported for 4D tensors.
    We leave 1D tensors unchanged.

    Otherwise, we raise an error if the tensor is not 4D.
    """
    if tensor.dim() < 2:
        # You didn't really mean it.
        return tensor

    if memory_format == torch.channels_last and tensor.dim() != 4:
        # You meant it, but it's not supported.
        raise ValueError("channels_last memory format is only supported for 4D tensors")

    return tensor.to(memory_format=memory_format)
