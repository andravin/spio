from math import prod

from .stats import Stats


class Conv2dStats(Stats):

    @property
    def output_macs(self):
        return self._size_output * self._accumulation_depth + self._bias_macs

    @property
    def grad_input_macs(self):
        return self._size_input * self._accumulation_depth

    @property
    def grad_weight_macs(self):
        return self._size_input * self._accumulation_depth

    @property
    def grad_bias_macs(self):
        return self._size_output / 2

    @property
    def output_bytes_read(self):
        return (self._size_input + self._size_weight + self._size_bias) * self.unit

    @property
    def output_bytes_written(self):
        return self._size_output * self.unit

    @property
    def grad_input_bytes_read(self):
        return (self._size_output + self._size_weight) * self.unit

    @property
    def grad_input_bytes_written(self):
        return self._size_input * self.unit

    @property
    def grad_weight_bytes_read(self):
        return (self._size_output + self._size_input) * self.unit

    @property
    def grad_weight_bytes_written(self):
        return self._size_weight * self.unit

    @property
    def grad_bias_bytes_read(self):
        return self._size_output * self.unit

    @property
    def grad_bias_bytes_written(self):
        return self._size_bias * self.unit

    @property
    def _size_input(self):
        return prod(self.params.input_shape)

    @property
    def _size_output(self):
        return prod(self.params.output_shape)

    @property
    def _size_weight(self):
        return prod(self.params.weight_shape)

    @property
    def _size_bias(self):
        return prod(self.params.bias_shape) if self.params.has_bias else 0

    @property
    def _accumulation_depth(self):
        return self.params.R * self.params.S * self.params.group_width

    @property
    def _bias_macs(self):
        return self._size_output / 2 if self.params.has_bias else 0
