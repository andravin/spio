"""Base class for kernel op and byte statistics."""

from typing import List

from .params import Params


class Stats:
    """Base class for kernel op and byte statistics.

    Statistics are computed for the calculations that produce
    the given output tensors. For example, to get statistics
    for the gradient of the input, set output_names to "grad_input".

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
    def macs(self) -> int:
        """The number of MACs in the calculations.

        Returns the total number of multiply-accumulates performed by
        all of the output tensor calculations.
        """
        return sum(
            getattr(self, f"{output_tensor}_macs")
            for output_tensor in self.output_names
        )

    @property
    def bytes_read(self) -> int:
        """The number of bytes read in the calculations."""
        return sum(
            getattr(self, f"{output_tensor}_bytes_read")
            for output_tensor in self.output_names
        )

    @property
    def bytes_written(self) -> int:
        """The number of bytes written in the calculations."""
        return sum(
            getattr(self, f"{output_tensor}_bytes_written")
            for output_tensor in self.output_names
        )

    @property
    def bytes(self) -> int:
        """Return the total number of bytes read and written."""
        return self.bytes_read + self.bytes_written

    @property
    def op_byte(self) -> float:
        """Return the number ops per byte."""
        return 2.0 * self.macs / self.bytes

    @property
    def accumulation_depths(self) -> int:
        """The accumulation depths of the calculations."""
        return [
            getattr(self, f"{output_tensor}_accumulation_depth")
            for output_tensor in self.output_names
        ]
