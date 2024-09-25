from pathlib import Path
import importlib.resources


def spio_include_path() -> str:
    return str(importlib.resources.files("spio").joinpath("include"))


def spio_src_path(filename: str) -> str:
    return str(importlib.resources.files("spio.src").joinpath(filename))


def spio_test_src_path(filename: str) -> str:
    return str(importlib.resources.files("spio.src_tests").joinpath(filename))


def spio_test_include_path() -> str:
    return str(importlib.resources.files("spio").joinpath("src_tests"))
