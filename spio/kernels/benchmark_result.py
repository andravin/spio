from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BenchmarkResult:
    """Dataclass that encapsulates the results of a benchmark run of a kernel configuration."""

    kernel_cls: Any = None
    kernel_kwargs: Dict[str, Any] = None
    device_desc: str = None
    params: Any = None
    config: Any = None
    kernel_idx: int = None
    time_ms: float = None
    tflop_s: float = None
    eff_bw_gb_s: float = None

    @property
    def time_s(self):
        return self.time_ms / 1e3

    def calculate_performance(self, gmacs: int, gbytes: int):
        self.tflop_s = gmacs / self.time_ms
        self.eff_bw_gb_s = gbytes / self.time_s
