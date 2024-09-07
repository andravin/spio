from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import torch
from frozendict import frozendict


# TODO combine initialize properties into a single field (empty, zero, one, none)
@dataclass
class ArgInfo:
    dtype: torch.dtype
    requires_grad: bool = False
    output: bool = False
    empty: bool = False
    zero: bool = False
    memory_format: torch.memory_format = torch.channels_last
    grad_of: str = None
    one: bool = False
    none: bool = False


@dataclass
class GradName:
    name: str
    grad_of: str


@dataclass
class KernelReflection:
    kernel_cls: Any
    arginfo: Dict[str, ArgInfo]
    kernel_args: List[str]
    function: staticmethod = None
    function_args: List[str] = None
    reference: staticmethod = None
    reference_args: List[str] = None
    stacking: List[Tuple[str, str]] = None
    is_grad: bool = False
    kernel_kwargs: defaultdict[frozendict] = field(default_factory=frozendict)

    def _shape(self, params, name):
        return getattr(params, f"{name}_shape")

    def _none(self):
        return None

    def _empty(self, params: Any, name: str, info: ArgInfo, device: str):
        return torch.empty(
            self._shape(params, name),
            dtype=info.dtype,
            device=device,
            memory_format=info.memory_format,
        )

    def _zero(self, params: Any, name: str, info: ArgInfo, device: str):
        t = torch.zeros(self._shape(params, name), dtype=info.dtype, device=device)
        return _to(t, memory_format=info.memory_format)

    def _ones(self, params: Any, name: str, info: ArgInfo, device: str):
        t = torch.ones(self._shape(params, name), dtype=info.dtype, device=device)
        return _to(t, memory_format=info.memory_format)

    def _randn(self, params: Any, name: str, info: ArgInfo, device: str):
        shape = self._shape(params, name)
        t = torch.randn(shape, dtype=info.dtype, device=device)
        return _to(t, memory_format=info.memory_format)

    def make_args(self, params, training=False, device="cuda"):
        args = dict()
        for name, info in self.arginfo.items():
            has_arg = _has_arg(params, name)
            if not has_arg or info.none:
                tensor = self._none()
            elif info.empty:
                tensor = self._empty(params, name, info, device)
            elif info.zero:
                tensor = self._zero(params, name, info, device)
            elif info.one:
                tensor = self._ones(params, name, info, device)
            else:
                tensor = self._randn(params, name, info, device)
            if info.requires_grad and training and has_arg:
                tensor.requires_grad = True
            args[name] = tensor
        return args

    def make_deltas(self, params, device="cuda"):
        args = dict()
        for name, info in self.arginfo.items():
            if info.output:
                args[name] = self._randn(params, name, info, device)
        return args

    def arrange_kernel_args(self, args) -> List[torch.Tensor]:
        return [args[name] for name in self.kernel_args]

    def arrange_function_args(self, args) -> List[torch.Tensor]:
        return [
            args[name] if args[name] is None or args[name].numel() > 0 else None
            for name in self.function_args
        ]

    def arrange_reference_args(self, args) -> List[torch.Tensor]:
        return [
            args[name] if args[name] is None or args[name].numel() > 0 else None
            for name in self.reference_args
        ]

    def zero_args(self, args):
        for name, tensor in args.items():
            info = self.arginfo[name]
            if info.zero:
                tensor.zero_()

    def get_grad_output_args(self, args):
        return [args[grad.name] for grad in self.grad_output_names]

    def get_differentiable_input_args(self, args):
        return [args[grad.grad_of] for grad in self.grad_input_names]

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
        function_args_lst = [self.arrange_function_args(args_lst[0])]
        for args in args_lst[1:]:
            function_args = self.arrange_function_args(args)
            function_args_lst.append(function_args[1:])
        return function_args_lst


reflection_kernel_registry = dict()

reflection_function_registry = dict()


@dataclass(frozen=True)
class KernelKey:
    kernel_cls: Any
    kernel_kwargs: defaultdict[frozendict] = field(default_factory=frozendict)


def register_reflection(reflection: KernelReflection):
    kernel_key = KernelKey(
        kernel_cls=reflection.kernel_cls,
        kernel_kwargs=frozendict(**reflection.kernel_kwargs),
    )
    reflection_kernel_registry[kernel_key] = reflection
    reflection_function_registry[reflection.function] = reflection


def get_kernel_reflection(kernel_cls: Any, **kernel_kwargs) -> KernelReflection:
    kernel_key = KernelKey(
        kernel_cls=kernel_cls, kernel_kwargs=frozendict(**kernel_kwargs)
    )
    return reflection_kernel_registry[kernel_key]


def get_function_reflection(function: Any) -> KernelReflection:
    return reflection_function_registry[function]


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
