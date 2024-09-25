import torch
from torch.fx import symbolic_trace

from ..kernels import Conv2dGw8Params
from ..layers import Conv2dGw8


spio_modules_classes = [Conv2dGw8]


def transform(model):
    """Transforms a PyTorch model by replacing matching modules with their Spio counterparts."""
    traced = symbolic_trace(model)
    modules_dict = dict(model.named_modules())
    for n in traced.graph.nodes:
        if n.op == "call_module" and n.target in modules_dict:
            module = modules_dict[n.target]
            for spio_module_class in spio_modules_classes:
                if spio_module_class.match(module):
                    spio_module = spio_module_class.from_torch_module(module)
                    _replace_module(traced, n, spio_module)
                    break
    traced.recompile()
    traced.delete_all_unused_submodules()
    traced.recompile()

    return traced


def scan_modules(model, *args):
    """Scans a PyTorch model and returns a list of parameters for every module that matches a Spio module."""
    traced = symbolic_trace(model)
    interpreter = _ScanInterpreter(traced)
    interpreter.run(*args)
    return interpreter.params_lst


def _replace_module(traced, n, spio_module):
    with traced.graph.inserting_after(n):
        spio_target = n.target + "_spio"
        traced.add_submodule(spio_target, spio_module)
        spio_node = traced.graph.call_module(spio_target, args=n.args, kwargs=n.kwargs)
        n.replace_all_uses_with(spio_node)
    traced.graph.erase_node(n)


class _ScanInterpreter(torch.fx.Interpreter):
    def __init__(self, model):
        self.modules_dict = dict(model.named_modules())
        super().__init__(model)
        self._params_lst = []

    def call_module(self, target, args, kwargs):
        params_lst = []
        if target in self.modules_dict:
            module = self.modules_dict[target]
            for spio_module_class in spio_modules_classes:
                if spio_module_class.match(module):
                    params = spio_module_class.Params.from_torch_module(module, *args)
                    self.params_lst.append(params)
                    break
        return super().call_module(target, args, kwargs)

    @property
    def params_lst(self):
        return self._params_lst
