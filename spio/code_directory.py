from tempfile import TemporaryDirectory
from pathlib import Path


class GenDirectory(TemporaryDirectory):
    def __init__(self):
        super().__init__(prefix="spio_")

    def __enter__(self):
        dir_name = super().__enter__()
        return Path(dir_name)
