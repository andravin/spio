from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class BenchmarkResult:
    """Dataclass that encapsulates the results of a benchmark run of a kernel configuration."""
    kernel_kwargs: Dict[str, Any] = None
    device_desc: str = None
    arch: str = None
    params: Any = None
    config: Any = None
    kernel_idx: int = None
    time_ms: float = None
    name: str = None
    stats: Any = None

    def __post_init__(self):
        """Calculate derived fields.

        Kernel gmacs and gbytes are calculated from the kernel_cls and params
        and are used to calculate performance."""
        if self.kernel_kwargs is None:
            self.kernel_kwargs = {}
        if self.stats is not None:
            self.gmacs = self.stats.macs / 1e9
            self.gbytes = (self.stats.bytes_read + self.stats.bytes_written) / 1e9
        if self.name is None:
            self.name = "Unknown"

    @property
    def time_s(self):
        return self.time_ms / 1e3

    @property
    def tflop_s(self):
        return self.gmacs / self.time_ms

    @property
    def eff_bw_gb_s(self):
        return self.gbytes / self.time_s
