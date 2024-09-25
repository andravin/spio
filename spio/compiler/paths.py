from pathlib import Path
import importlib.resources


def spio_include_path() -> str:
    with importlib.resources.files("spio").joinpath("include") as path:
        return str(path)


def spio_src_path(filename: str) -> str:
    with importlib.resources.files("spio.src").joinpath(filename) as path:
        return str(path)


def spio_test_src_path(filename: str) -> str:
    with importlib.resources.files("spio.src_tests").joinpath(filename) as path:
        return str(path)

def spio_test_include_path() -> str:
    with importlib.resources.files("spio").joinpath("src_tests") as path:
        return str(path)
