"""Function to load a Params set from a resource file."""

from typing import List, Type, TYPE_CHECKING
from dataclasses import dataclass


from importlib_resources import files as importlib_resources_files

from .parse_dataclass import parse_dataclass


if TYPE_CHECKING:
    from ..kernels import Params


def _load_dataclasses_from_resource(
    resource_name: str, dataclasses: List[Type[dataclass]] = None
) -> List[dataclass]:
    """Load a list of dataclasses from a resource file.

    The resource file is located at spio.data.<resource_name>.dat.

    Each line in the resource file should be a valid Python expression
    corresponding to a dataclass instance. For example, the following
    lines are valid expressions for a dataclass with fields
    `a: int`, `b: str`, and `c: float`:

    ```
    Dataclass(a=1, b="hello", c=3.14)
    Dataclass(a=2, b="world", c=2.71)
    ```

    Parameters
    ----------
    resource_name : str
        The name of the resource file.
    dataclasses : List[dataclass]
        A list of dataclasses to use for parsing the expressions.

    Returns
    -------
    List[dataclass]
        A list of dataclass instances parsed from the resource file.
    """
    if dataclasses is None:
        return []
    dataclasses = {d.__name__: d for d in dataclasses}
    params_lst = []
    with importlib_resources_files("spio.src_tests").joinpath(resource_name).open(
        "r"
    ) as f:
        for line in f:
            if line:
                params = parse_dataclass(line, dataclasses=dataclasses)
                if params is not None:
                    params_lst.append(params)
    return params_lst


def load_parameter_set(params_cls: Type["Params"] = None):
    """Load the parameter set for the given parameter dataclass.

    Paremeter sets are located in spio.data.<data_cls>.dat. They
    are used to train performance models and generate test cases.

    Parameters
    ----------
    params_cls : dataclass
        The dataclass for the parameter set.

    Returns
    -------
    List[dataclass]
        A list of parameter sets for the given dataclass.
    """
    resource = f"{params_cls.__name__}.dat"
    return _load_dataclasses_from_resource(resource, [params_cls])
