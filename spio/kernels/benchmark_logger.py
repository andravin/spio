import os

from .benchmark_formatter import BenchmarkResultCompactFormat, BenchmarkResultFullFormat
from .benchmark_result import BenchmarkResult

TRUTHS = ["true", "1", "yes", "y", "t"]
default_enable_logger = os.environ.get("SPIO_LOGGER", "False").lower() in TRUTHS


class BenchmarkLogger:

    def __init__(self, **kwargs):
        self.configure(**kwargs)

    def configure(
        self,
        full_format: bool = True,
        header: bool = True,
        best: bool = True,
        sort: bool = True,
        header_once: bool = False,
        enable=False,
    ):
        format_cls = (
            BenchmarkResultFullFormat if full_format else BenchmarkResultCompactFormat
        )
        self._result_format = format_cls()
        self.header = header
        self.best = best
        self.sort = sort
        self.once = True
        self.header_once = header_once
        self.enable = enable

    def log_best(self, result: BenchmarkResult):
        if self.best and self.enable:
            msg = self._result_format.result(result, best=True)
            print(msg)

    def log_results(self, results):
        if self.enable:
            show_header = self.header and (not self.header_once or self.once)
            msg = self._result_format.results(
                results, sort=self.sort, header=show_header
            )
            print(msg)
            self.once = False


benchmark_logger = BenchmarkLogger(enable=default_enable_logger)
