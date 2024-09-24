from pathlib import Path
import importlib.resources


def spio_include_path() -> str:
    with importlib.resources.path("spio", "include") as path:
        return str(path)


def spio_src_path(filename: str) -> str:
    with importlib.resources.path("spio.src", filename) as path:
        return str(path)


def spio_test_src_path(filename: str) -> str:
    with importlib.resources.path("spio.src_tests", filename) as path:
        return str(path)

def spio_test_include_path() -> str:
    with importlib.resources.path("spio", "src_tests") as path:
        return str(path)
