from typing import List
from dataclasses import dataclass, asdict
from functools import partial

import pandas as pd


def params_and_configs_to_dataframe(params, configs):
    df_params = pd.DataFrame([params])
    df_params = df_params.add_prefix("Params_")
    df_params = _expand_dataframe_tuples(df_params)

    df_configs = pd.DataFrame(configs)
    df_configs = df_configs.add_prefix("Config_")
    df_configs = _expand_dataframe_tuples(df_configs)

    df = pd.concat([df_params] * len(df_configs), ignore_index=True).join(df_configs.reset_index(drop=True))

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
