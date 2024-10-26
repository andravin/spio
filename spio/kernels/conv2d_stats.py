"""Memory and compute statistics for a 2D convolution kernel."""

from math import prod

from .stats import Stats


class Conv2dStats(Stats):
    """Memory and compute statistics for a 2D convolution kernel."""

    @property
    def output_macs(self):
        """The number of MACs performed during the forward pass."""
        return self._size_output * self._accumulation_depth + self._bias_macs

    @property
    def grad_input_macs(self):
        """The number of MACs performed for the input gradient."""
        return self._size_input * self._accumulation_depth

    @property
    def grad_weight_macs(self):
        """Return the number of MACs for the weights gradient."""
        return self._size_input * self._accumulation_depth

    @property
    def grad_bias_macs(self):
        """The number of MACs for the bias gradient."""
        return self._size_output / 2

    @property
    def output_bytes_read(self):
        """The number of bytes read during the forward pass."""
        return (self._size_input + self._size_weight + self._size_bias) * self.unit

    @property
    def output_bytes_written(self):
        """The number of bytes written during the forward pass."""
        return self._size_output * self.unit

    @property
    def grad_input_bytes_read(self):
        """The bytes read for the inputs gradient calculation."""
        return (self._size_output + self._size_weight) * self.unit

    @property
    def grad_input_bytes_written(self):
        """The bytes written for the input gradient calculation."""
        return self._size_input * self.unit

    @property
    def grad_weight_bytes_read(self):
        """The bytes read for the weights gradient calculation."""
        return (self._size_output + self._size_input) * self.unit

    @property
    def grad_weight_bytes_written(self):
        """The bytes written for the weights gradient calculation."""
        return self._size_weight * self.unit

    @property
    def grad_bias_bytes_read(self):
        """The bytes read for the bias gradient calculation."""
        return self._size_output * self.unit

    @property
    def grad_bias_bytes_written(self):
        """The bytes written for the bias gradient calculation."""
        return self._size_bias * self.unit

    @property
    def output_accumulation_depth(self):
        """The depth of accumulation in the output tensor."""
        return self._accumulation_depth

    @property
    def grad_input_accumulation_depth(self):
        """The depth of accumulation of the input gradient."""
        return self._accumulation_depth

    @property
    def grad_weight_accumulation_depth(self):
        """The depth of accumulation of the weight gradient."""
        return self.params.n * self.params.h * self.params.w

    @property
    def grad_bias_accumulation_depth(self):
        """The depth of accumulation of the bias gradient."""
        return self.params.n * self.params.h * self.params.w

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

        This is the number of multiply-accumulates (MACs) performed per
        output element.
        """
        return self.params.r * self.params.s * self.params.group_width

    @property
    def _bias_macs(self):
        """The number of multiply-accumulates for the bias vector.

        We count an addition as 0.5 MACs.
        """
        return self._size_output / 2 if self.params.has_bias else 0
