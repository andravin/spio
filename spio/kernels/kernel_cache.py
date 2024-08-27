from .auto_tune import auto_tune


class KernelCache:
    def __init__(self):
        self._cache = {}

    def get(self, kernel_cls, params, args, config=None, **kwargs):
        if config is None:
            if params not in self._cache:
                kernel, _, _ = auto_tune(kernel_cls, params, args, **kwargs)
                self._cache[params] = kernel
            return self._cache[params]
        else:
            return kernel_cls(params, config=config, **kwargs)
