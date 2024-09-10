from typing import List

from .benchmark_result import BenchmarkResult


class BenchmarkResultFormat:
    def __init__(self, delim: str = None):
        self.delim = delim

    def results(
        self, results: List[BenchmarkResult], sort=False, header=True, reverse=True
    ):
        if sort:
            results = sorted(results, key=lambda x: x.time_ms, reverse=reverse)
        txt = ""
        if header:
            txt += self.header()
        txt += "".join([self.result(result) for result in results])
        return txt


class BenchmarkResultCompactFormat(BenchmarkResultFormat):
    def header(self):
        config = "Config"
        time = "Time[ms]"
        tflop_s = "TFLOP/s"
        eff_bw_gb_s = "Eff BW[GB/s]"
        idx = "Idx"
        if self.delim is not None:
            d = self.delim
            return f"{idx:s}{d}{config:s}{d}{time:s}{d}{tflop_s:s}{d}{eff_bw_gb_s:s}\n"
        else:
            return f"[{idx:5s}] {config:90s} {time:8s} {tflop_s:8s} {eff_bw_gb_s:8s}\n"

    def result(self, result: BenchmarkResult, best=False):
        idx = str(result.kernel_idx)
        if best:
            idx = f"{idx} *"
        if self.delim is not None:
            d = self.delim
            return f"{idx:s}{d}{str(result.config):s}{d}{result.time_ms:.3f}{d}{result.tflop_s:.3f}{d}{result.eff_bw_gb_s:.3f}\n"
        else:
            return f"[{idx:5s}] {str(result.config):90s} {result.time_ms:8.3f} {result.tflop_s:8.3f} {result.eff_bw_gb_s:8.3f}\n"


class BenchmarkResultFullFormat(BenchmarkResultFormat):
    def header(self):
        kernel_cls = "Kernel"
        kernel_kwargs = "Kernel_Kwargs"
        device_desc = "Device"
        params = "Params"
        idx = "Idx"
        config = "Config"
        time = "Time[ms]"
        tflop_s = "TFLOP/s"
        eff_bw_gb_s = "Eff BW[GB/s]"
        arch = "Arch"
        if self.delim is not None:
            d = self.delim
            return f"{kernel_cls:s}{d}{kernel_kwargs:s}{d}{device_desc:s}{d}{arch:s}{d}{params:s}{d}{idx:s}{d}{config:s}{d}{time:s}{d}{tflop_s:s}{d}{eff_bw_gb_s:s}\n"
        else:
            return f"{kernel_cls:80s} {kernel_kwargs:20s} {device_desc:30s} {arch:5s} {params:100s} {idx:>5s}  {config:90s} {time:>8s} {tflop_s:>8s} {eff_bw_gb_s:>8s}\n"

    def result(self, result: BenchmarkResult, best=False):
        idx = str(result.kernel_idx)
        if best:
            idx = f"{idx} *"
        if self.delim is not None:
            d = self.delim
            return f"{result.name:s}{d}{str(result.kernel_kwargs):s}{d}{result.device_desc:s}{d}{result.arch}{d}{str(result.params):s}{d}{idx:s}{d}{str(result.config):s}{d}{result.time_ms:.3f}{d}{result.tflop_s:.3f}{d}{result.eff_bw_gb_s:.3f}\n"
        else:
            return f"{result.name:80s} {str(result.kernel_kwargs):20s} {result.device_desc:30s} {result.arch:5s} {str(result.params):100s} {idx:>5s}  {str(result.config):90s} {result.time_ms:8.3f} {result.tflop_s:8.3f} {result.eff_bw_gb_s:8.3f}\n"
