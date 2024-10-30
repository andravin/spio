"""Functions to transform and scan PyTorch modules with Spio modules."""

from typing import List

import torch
from torch.fx import symbolic_trace

from ..layers import Conv2dGw8
from ..util import get_full_name, logger_enabled
from ..kernels import Params


spio_modules_classes = [Conv2dGw8]


def transform(model: torch.nn.Module):
    """Transforms a PyTorch model to use Spio modules.

    Replaces any matching modules with their Spio counterparts.
    """
    return _transform(model)[0]


def _transform(model: torch.nn.Module):
    """Transform the model.

    Returns the transformed model and the number of modules replaced.
    """
    traced = symbolic_trace(model)
    num_replacements = 0
    modules_dict = dict(model.named_modules())
    nodes_to_replace = []

    # Collect nodes to be replaced
    for n in traced.graph.nodes:
        if n.op == "call_module" and n.target in modules_dict:
            module = modules_dict[n.target]
            for spio_module_class in spio_modules_classes:
                if spio_module_class.match(module):
                    nodes_to_replace.append(
                        (n, spio_module_class.from_torch_module(module))
                    )
                    break

    # Replace collected nodes
    for n, spio_module in nodes_to_replace:
        _replace_module(traced, n, spio_module)
        if logger_enabled:
            before = get_full_name(modules_dict[n.target].__class__)
            after = spio_module.__class__.__module__ + "." + str(spio_module)
            print(f"spio: replaced a {before} module with {after}")
        num_replacements += 1

    traced.delete_all_unused_submodules()
    traced.recompile()

    return traced, num_replacements


def scan_modules(model, *args) -> List[Params]:
    """Returns a list of Params for matched Spio modules.

    Scans a PyTorch model and returns a list of Params for the matched
    Spio modules.
    """
    traced = symbolic_trace(model)
    interpreter = _ScanInterpreter(traced)
    interpreter.run(*args)
    return interpreter.params_lst


def _replace_module(traced, n, spio_module):
    # Directly set the new Spio module in the model's named_modules dictionary
    if "." in n.target:
        parent_name, module_name = n.target.rsplit(".", 1)
        parent_module = dict(traced.named_modules())[parent_name]
    else:
        parent_module = traced
        module_name = n.target
    setattr(parent_module, module_name, spio_module)

    # Replace the node in the graph with the new Spio module
    with traced.graph.inserting_after(n):
        spio_node = traced.graph.call_module(n.target, args=n.args, kwargs=n.kwargs)
        n.replace_all_uses_with(spio_node)
    traced.graph.erase_node(n)


class _ScanInterpreter(torch.fx.Interpreter):
    """Interpreter for scanning modules in a PyTorch model."""

    def __init__(self, model):
        self.modules_dict = dict(model.named_modules())
        super().__init__(model)
        self._params_lst = []

    def call_module(self, target, args, kwargs):
        if target in self.modules_dict:
            module = self.modules_dict[target]
            for spio_module_class in spio_modules_classes:
                if spio_module_class.match(module):
                    params = spio_module_class.Params.from_torch_module(module, *args)
                    self.params_lst.append(params)
                    break
        return super().call_module(target, args, kwargs)

    @property
    def params_lst(self) -> List[Params]:
        """Return the list of parameters recorded so far."""
        return self._params_lst
