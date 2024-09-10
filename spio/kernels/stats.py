class Stats:
    def __init__(self, params=None, unit=2, output_names=None):
        self.params = params
        self.unit = unit
        if isinstance(output_names, str):
            output_names = [output_names]
        self.output_names = output_names

    @property
    def macs(self):
        return sum(
            [
                getattr(self, f"{output_tensor}_macs")
                for output_tensor in self.output_names
            ]
        )

    @property
    def bytes_read(self):
        return sum(
            [
                getattr(self, f"{output_tensor}_bytes_read")
                for output_tensor in self.output_names
            ]
        )

    @property
    def bytes_written(self):
        return sum(
            [
                getattr(self, f"{output_tensor}_bytes_written")
                for output_tensor in self.output_names
            ]
        )

    @property
    def bytes(self):
        return self.bytes_read + self.bytes_written

    @property
    def op_byte(self):
        return 2.0 * self.macs / self.bytes
