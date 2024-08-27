from .benchmark import benchmark


class KernelCache:
    def __init__(self):
        self._cache = {}

    def get(self, kernel_cls, params, args, config=None, **kwargs):
        if config is None:
            if params not in self._cache:
                best = benchmark(kernel_cls, params, [args], **kwargs)
                self._cache[params] = best.kernel
            return self._cache[params]
        else:
            return kernel_cls(params, config=config, **kwargs)
