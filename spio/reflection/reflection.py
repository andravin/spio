"""
This module provides a reflection system for kernels and functions.

The reflection system is used to define the arguments and properties of a kernel or function.
Tests and benchmarks use the reflection system to generate arguments for the kernel or function.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import torch

from .arg_info import ArgInfo, Init


@dataclass
class GradName:
    """Encode the relationship between a gradient argument and the argument it is the gradient of."""

    name: str
    grad_of: str


@dataclass
class Reflection:
    """Reflection information for a kernel or function.

    The reflection system is used to define the arguments and properties of a kernel, function, or layer.
    Tests and benchmarks use the reflection system to generate arguments.
    """

    arginfo: Dict[str, ArgInfo]

    kernel_name: str = None

    function: staticmethod = None

    layer_cls: Any = None

    args: List[str] = field(default_factory=list)
    
    kwargs: Dict[str, Any] = field(default_factory=dict)

    kernel_outputs: List[str] = field(default_factory=list)

    stacking: List[Tuple[str, str]] = None

    reference: staticmethod = None

    stats: Any = None

    params: Any = None

    prefix_op_constructor: staticmethod = None

    get_function_kwargs: staticmethod = None

    from_layer: staticmethod = None

    memory_format = torch.channels_last

    ignore_params: List[str] = None

    def __post_init__(self):
        for name, arg in self.arginfo.items():
            if arg.grad_of is not None:
                assert arg.grad_of in self.arginfo
            arg.name = name

    def make_args(self, params, training=False, device="cuda"):
        args = dict()
        for name, info in self.arginfo.items():
            args[name] = info.make_arg(params, training=training, device=device)
        return args

    def make_grad_outputs(self, params, device="cuda"):
        args = dict()
        for name, info in self.arginfo.items():
            if info.output:
                args[f"grad_{name}"] = info._randn(params, device)
        return args

    def arrange_args(self, args) -> List[torch.Tensor]:
        return [
            args[name] if args[name] is None or args[name].numel() > 0 else None
            for name in self.args
        ]

    def init_args(self, args):
        for name, tensor in args.items():
            info = self.arginfo[name]
            info.initialize(tensor)

    def init_zeros(self, args):
        for name, tensor in args.items():
            info = self.arginfo[name]
            if info.init == Init.ZERO:
                tensor.zero_()

    def get_grad_output_args(self, args):
        return [args[grad.name] for grad in self.grad_output_names]

    def get_differentiable_input_args(self, args):
        return [
            args[name]
            for name in self.get_differentiable_input_names(args)
            if args[name] is not None
        ]

    def get_differentiable_input_names(self):
        return [
            info.name
            for info in self.arginfo.values()
            if info.requires_grad and not info.output
        ]

    def get_arg_name_from_gradient(self, gradient_name):
        for name, info in self.arginfo.items():
            if info.name == gradient_name:
                return info.grad_of
        raise ValueError("No argument found for gradient name: " + gradient_name)

    @property
    def output_names(self) -> List[str]:
        return [name for name, info in self.arginfo.items() if info.output]

    @property
    def grad_input_names(self) -> List[GradName]:
        """Return the names of the gradients that this kernel computes."""
        return [
            GradName(name=name, grad_of=info.grad_of)
            for name, info in self.arginfo.items()
            if info.grad_of is not None and not self.arginfo[info.grad_of].output
        ]

    @property
    def grad_output_names(self) -> List[GradName]:
        """Return the names of the gradients of the outputs."""
        return [
            GradName(name, info.grad_of)
            for name, info in self.arginfo.items()
            if info.grad_of is not None and self.arginfo[info.grad_of].output
        ]

    @property
    def stackable(self) -> bool:
        return self.stacking is not None

    def make_stacked_args(
        self, params: Any, depth: int = 1, training=False, device="cuda"
    ):
        if not self.stackable:
            raise ValueError("No stacking defined for this kernel.")
        args = self.make_args(params, training=training, device=device)
        args_lst = [args]
        for d in range(depth - 1):
            next_args = self.make_args(params, training=training, device=device)
            for name, next_name in self.stacking:
                next_args[next_name] = args[name]
            args_lst.append(next_args)
            args = next_args
        return args_lst

    def arrange_stacked_args_for_function(self, args_lst):
        """TODO: modify this to work when the function has more than one input."""
        function_args_lst = [self.arrange_args(args_lst[0])]
        for args in args_lst[1:]:
            function_args = self.arrange_args(args)
            function_args_lst.append(function_args[1:])
        return function_args_lst

    @property
    def kernel_is_backprop(self):
        """Return true if this reflection is for a backprop kernel.

        A backprop kernel is a kernel that computes the gradient of a function.
        A kernel is backprop kernel if any of its arguments start with "grad_".
        """
        return self.kernel_name is not None and any(
            [arg.startswith("grad_") for arg in self.args]
        )


reflection_kernel_registry = dict()

reflection_function_registry = dict()

reflection_layer_registry = dict()


def register_reflection(reflection: Reflection):
    if _more_than_one_not_none(
        reflection.kernel_name, reflection.function, reflection.layer_cls
    ):
        raise ValueError(
            f"Reflection must have exactly one of kernel_cls, layer_cls, or function: {reflection}"
        )
    if reflection.kernel_name is not None:
        reflection_kernel_registry[reflection.kernel_name] = reflection
    elif reflection.function is not None:
        reflection_function_registry[reflection.function] = reflection
    elif reflection.layer_cls is not None:
        reflection_layer_registry[reflection.layer_cls] = reflection


def get_kernel_reflection(kernel_name: str) -> Reflection:
    return reflection_kernel_registry[kernel_name]


def get_function_reflection(function: Any) -> Reflection:
    return reflection_function_registry[function]


def get_layer_reflection(layer_cls: Any) -> Reflection:
    return reflection_layer_registry[layer_cls]


def _more_than_one_not_none(*args):
    return sum(arg is not None for arg in args) != 1
