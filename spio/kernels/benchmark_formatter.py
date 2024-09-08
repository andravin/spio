from typing import List

from .benchmark_result import BenchmarkResult


class BenchmarkResultFormat:
    def results(self, results: List[BenchmarkResult], sort=False, header=True, reverse=True):
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
        return f"[{idx:5s}] {config:90s} {time:8s} {tflop_s:8s} {eff_bw_gb_s:8s}\n"

    def result(self, result: BenchmarkResult, best=False):
        idx = str(result.kernel_idx)
        if best:
            idx = f"{idx} *"
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
        return f"{kernel_cls:30s} {kernel_kwargs:20s} {device_desc:30s} {params:90s} {idx:5s} {config:90s} {time:8s} {tflop_s:8s} {eff_bw_gb_s:8s}\n"

    def result(self, result: BenchmarkResult, best=False):
        idx = str(result.kernel_idx)
        if best:
            idx = f"{idx} *"
        return f"{result.name:30s} {str(result.kernel_kwargs):20s} {str(result.device_desc):30s} {str(result.params):90s} {idx:5s} {str(result.config):90s} {result.time_ms:8.3f} {result.tflop_s:8.3f} {result.eff_bw_gb_s:8.3f}\n"
