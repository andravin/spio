import torch
from torch.fx import symbolic_trace

from spio.layers import Conv2dGw8


def transform(model):
    traced = symbolic_trace(model)
    patterns = set([torch.nn.functional.conv2d])
    modules_dict = dict(model.named_modules())
    for n in traced.graph.nodes:
        if n.op == "call_module" and n.target in modules_dict:
            module = modules_dict[n.target]
            if isinstance(module, torch.nn.Conv2d):
                if Conv2dGw8.match(module):
                    spio_module = Conv2dGw8.from_conv2d(module)
                    _replace_module(traced, n, spio_module)
                    print(
                        f"replace: target: {n.target} op:{n.op} module: {module} with {spio_module}"
                    )
    traced.recompile()
    traced.delete_all_unused_submodules()
    traced.recompile()

    return traced


def _replace_module(traced, n, spio_module):
    with traced.graph.inserting_after(n):
        spio_target = n.target + "_spio"
        traced.add_submodule(spio_target, spio_module)
        spio_node = traced.graph.call_module(spio_target, args=n.args, kwargs=n.kwargs)
        n.replace_all_uses_with(spio_node)
    traced.graph.erase_node(n)
