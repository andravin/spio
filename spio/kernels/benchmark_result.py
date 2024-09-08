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
    name: str = None

    def __post_init__(self):
        """Calculate derived fields.

        Kernel gmacs and gbytes are calculated from the kernel_cls and params
        and are used to calculate performance."""
        if self.kernel_kwargs is None:
            self.kernel_kwargs = {}
        if self.kernel_cls is not None:
            macs = self.kernel_cls.macs(self.params, **self.kernel_kwargs)
            bytes_read = self.kernel_cls.bytes_read(self.params, **self.kernel_kwargs)
            bytes_written = self.kernel_cls.bytes_written(
                self.params, **self.kernel_kwargs
            )
            bytes = bytes_read + bytes_written
            self.gmacs = macs / 1e9
            self.gbytes = bytes / 1e9
        if self.name is None:
            if self.kernel_cls is not None:
                self.name = self.kernel_cls.__name__
            else:
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
