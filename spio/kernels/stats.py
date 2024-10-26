"""Base class for kernel op and byte statistics."""

from typing import List

from .params import Params


class Stats:
    """Base class for kernel op and byte statistics.

    Args:
        params: Parameters for the layer.
        unit: number of bytes per tensor element.
        output_names: Names of the output tensors.
    """

    def __init__(
        self, params: Params = None, unit: int = 2, output_names: List[str] = None
    ):
        self.params = params
        self.unit = unit
        if isinstance(output_names, str):
            output_names = [output_names]
        self.output_names = output_names

    @property
    def macs(self):
        """Return the number of multicply-accumulates performed for the output tensors."""
        return sum(
            getattr(self, f"{output_tensor}_macs")
            for output_tensor in self.output_names
        )

    @property
    def bytes_read(self):
        """Return the number of bytes read while computing the output tensors."""
        return sum(
            getattr(self, f"{output_tensor}_bytes_read")
            for output_tensor in self.output_names
        )

    @property
    def bytes_written(self):
        """Return the number of bytes written while computing the output tensors."""
        return sum(
            getattr(self, f"{output_tensor}_bytes_written")
            for output_tensor in self.output_names
        )

    @property
    def bytes(self):
        """Return the total number of bytes read and written."""
        return self.bytes_read + self.bytes_written

    @property
    def op_byte(self):
        """Return the number ops per byte."""
        return 2.0 * self.macs / self.bytes

    @property
    def accumulation_depths(self):
        """Return the accumulation depths used by the compute for each output tensor."""
        return [
            getattr(self, f"{output_tensor}_accumulation_depth")
            for output_tensor in self.output_names
        ]
