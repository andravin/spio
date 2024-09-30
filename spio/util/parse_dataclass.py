from typing import List, Type
from dataclasses import dataclass, asdict
from functools import partial
import importlib.resources

import pandas as pd


def load_parameter_set(params_cls: dataclass = None):
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
    return load_dataclasses_from_resource(resource, [params_cls])


def load_dataclasses_from_resource(
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
    with importlib.resources.files("spio.data").joinpath(resource_name).open("r") as f:
        for line in f:
            if line:
                params = parse_dataclass(line, dataclasses=dataclasses)
                if params is not None:
                    params_lst.append(params)
    return params_lst


def params_and_configs_to_dataframe(params, configs):
    df_params = pd.DataFrame([params])
    # FIXME These fields from Conv2dGw8Params are really constants.
    # FIXME Perhaps there should be an "ignore_fields" argument.
    df_params.drop("group_width", axis=1, inplace=True) 
    df_params.drop("stride", axis=1, inplace=True) 
    df_params = df_params.add_prefix("Params_")
    df_params = _expand_dataframe_tuples(df_params)

    df_configs = pd.DataFrame(configs)
    df_configs = df_configs.add_prefix("Config_")
    df_configs = _expand_dataframe_tuples(df_configs)

    df = pd.concat([df_params] * len(df_configs), ignore_index=True).join(
        df_configs.reset_index(drop=True)
    )

    return df


def _expand_tuple_fields(row):
    expanded_row = {}
    for key, value in row.items():
        if isinstance(value, tuple) and all(isinstance(i, int) for i in value):
            for i, elem in enumerate(value):
                expanded_row[f"{key}_{i}"] = elem
        else:
            expanded_row[key] = value
    return expanded_row


def _expand_dataframe_tuples(df):
    expanded_rows = df.apply(lambda row: _expand_tuple_fields(row), axis=1)
    expanded_df = pd.DataFrame(expanded_rows.tolist())
    return expanded_df


def dataclass_to_series(dataclass_obj):
    data = {}
    for key, value in dataclass_obj.__dict__.items():
        if isinstance(value, tuple):
            for i, elem in enumerate(value):
                data[f"{key}_{i}"] = elem
        else:
            data[key] = value
    return pd.Series(data)


def parse_dataclass(expr: str, dataclasses=None) -> dataclass:
    expr = expr.strip()
    if expr:
        try:
            return eval(expr, dataclasses)
        except (SyntaxError, NameError) as e:
            raise ValueError(f"Failed to parse line '{expr}': {e}")
    return None


def import_dataclass_column(df, col, dataclasses):
    parse_these_dataclasses = partial(parse_dataclass, dataclasses=dataclasses)
    df[col] = df[col].apply(parse_these_dataclasses)
    attributes_df = df[col].apply(dataclass_to_series)

    # Prefix the attribute column names to avoid conflicts
    attributes_df = attributes_df.add_prefix(f"{col}_")

    # Join the new attributes DataFrame with the original DataFrame
    df = pd.concat([df, attributes_df], axis=1)

    # Drop the original dataclass column if no longer needed
    df.drop(columns=[col], inplace=True)
    return df


# Combine Params and Configs
def _combine_params_config(params, config):
    params_dict = asdict(params)
    config_dict = asdict(config)
    combined_dict = {**params_dict, **config_dict}
    return combined_dict
