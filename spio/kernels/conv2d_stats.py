from math import prod

from .stats import Stats


class Conv2dStats(Stats):
    """Memory and compute statistics for a 2D convolution kernel."""

    @property
    def output_macs(self):
        """Return the number of MACs performed during the forward pass."""
        return self._size_output * self._accumulation_depth + self._bias_macs

    @property
    def grad_input_macs(self):
        """Return the number of MACs performed during the backward pass w.r.t. input."""
        return self._size_input * self._accumulation_depth

    @property
    def grad_weight_macs(self):
        """Return the number of MACs performed during the backward pass w.r.t. weight."""
        return self._size_input * self._accumulation_depth

    @property
    def grad_bias_macs(self):
        """Return the number of MACs performed during the backward pass w.r.t. bias."""
        return self._size_output / 2

    @property
    def output_bytes_read(self):
        """Return the number of bytes read during the forward pass."""
        return (self._size_input + self._size_weight + self._size_bias) * self.unit

    @property
    def output_bytes_written(self):
        """Return the number of bytes written during the forward pass."""
        return self._size_output * self.unit

    @property
    def grad_input_bytes_read(self):
        """Return the number of bytes read during the backward pass w.r.t. input."""
        return (self._size_output + self._size_weight) * self.unit

    @property
    def grad_input_bytes_written(self):
        """Return the number of bytes written during the backward pass w.r.t. input."""
        return self._size_input * self.unit

    @property
    def grad_weight_bytes_read(self):
        """Return the number of bytes read during the backward pass w.r.t. weight."""
        return (self._size_output + self._size_input) * self.unit

    @property
    def grad_weight_bytes_written(self):
        """Return the number of bytes written during the backward pass w.r.t. weight."""
        return self._size_weight * self.unit

    @property
    def grad_bias_bytes_read(self):
        """Return the number of bytes read during the backward pass w.r.t. bias."""
        return self._size_output * self.unit

    @property
    def grad_bias_bytes_written(self):
        """Return the number of bytes written during the backward pass w.r.t. bias."""
        return self._size_bias * self.unit

    @property
    def _size_input(self):
        """Return the number of elements in the input tensor."""
        return prod(self.params.input_shape)

    @property
    def _size_output(self):
        """Return the number of elements in the output tensor."""
        return prod(self.params.output_shape)

    @property
    def _size_weight(self):
        """Return the number of elements in the weight tensor."""
        return prod(self.params.weight_shape)

    @property
    def _size_bias(self):
        """Return the number of elements in the bias vector."""
        return prod(self.params.bias_shape) if self.params.has_bias else 0

    @property
    def _accumulation_depth(self):
        """Return depth of accumulation in the output tensor.

        This is the number of multiply-accumulates (MACs) performed per output element.
        """
        return self.params.R * self.params.S * self.params.group_width

    @property
    def _bias_macs(self):
        """Return the number of multiply-accumulates performed by the bias vector.
        
        We count an addition as 0.5 MACs.
        """
        return self._size_output / 2 if self.params.has_bias else 0
