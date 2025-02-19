"""Memory and compute statistics for a multi-layer perceptron (MLP) kernel."""

from math import prod

from .stats import Stats


class MlpStats(Stats):
    """Memory and compute statistics for a multi-layer perceptron (MLP) kernel."""

    @property
    def output_macs(self):
        """The number of MACs performed during the forward pass."""
        return self.exp_macs + self.prj_macs

    @property
    def exp_macs(self):
        """The number of MACs performed during the expansion layer."""
        return self.params.x * self.params.c * self.params.r

    @property
    def prj_macs(self):
        """The number of MACs performed during the projection layer."""
        return self.params.x * self.params.r * self.params.k

    @property
    def output_bytes_read(self):
        """The number of bytes read during the forward pass."""
        return (
            self._size_input
            + self._size_exp_weight
            + self._size_exp_bias
            + self._size_prj_weight
            + self._size_prj_bias
        ) * self.unit

    @property
    def output_bytes_written(self):
        """The number of bytes written during the forward pass."""
        return self._size_output * self.unit

    @property
    def output_accumulation_depth(self):
        """The accumulation depth of the output calculation.

        TODO: Accumulation depths is not a valid way to estimate the accuracy
        of two consecutive layers. Allow stats to provide a custom accuracy estimate?
        """
        return self.params.c + self.params.r

    @property
    def _size_output(self):
        return prod(self.params.output_shape)

    @property
    def _size_input(self):
        return prod(self.params.input_shape)

    @property
    def _size_exp_weight(self):
        return prod(self.params.exp_weight_shape)

    @property
    def _size_exp_bias(self):
        return prod(self.params.exp_bias_shape) if self.params.bias else 0

    @property
    def _size_prj_weight(self):
        return prod(self.params.prj_weight_shape)

    @property
    def _size_prj_bias(self):
        return prod(self.params.prj_bias_shape) if self.params.bias else 0
