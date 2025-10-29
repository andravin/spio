"""Memory and compute statistics for a 2D layer normalization kernel."""

from math import prod

from .stats import Stats


class LayerNorm2dStats(Stats):
    """Memory and compute statistics for a 2D layer normalization kernel."""

    @property
    def output_macs(self):
        """The number of MACs performed during the forward pass."""
        raise self._size_output * self._macs_per_output

    @property
    def output_bytes_read(self):
        """The number of bytes read during the forward pass."""
        return (self._size_input + self._size_weights + self._size_bias) * self.unit

    @property
    def output_bytes_written(self):
        """The number of bytes written during the forward pass."""
        return self._size_output * self.unit

    @property
    def _size_input(self):
        return prod(self.params.input_shape)

    @property
    def _size_output(self):
        return prod(self.params.output_shape)

    @property
    def _size_weights(self):
        return prod(self.params.weight_shape) if self.params.elementwise_affine else 0

    @property
    def _size_bias(self):
        return prod(self.params.bias_shape) if self.params.has_bias else 0

    @property
    def _macs_per_output(self):
        return 3 if self.params.elementwise_affine else 2

    @property
    def output_accumulation_depth(self):
        """The accumulation depth of the output calculation."""
        return self.params.c
