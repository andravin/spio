"""Train a performance model for a given kernel."""

import argparse
import os
import glob
from pathlib import Path
from functools import partial

import xgboost as xgb

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from tabulate import tabulate

from spio.kernels import (
    Conv2dGw8Params,
    Conv2dGw8Config,
    Conv2dGw8WgradConfig,
    get_device_performance_model_file_name,
    PERFORMANCE_MODEL_EXTENSION,
)
from spio.util.parse_dataclass import parse_dataclass
from spio.reflection import get_kernel_reflection


PARAMS_CLASSES = {"Conv2dGw8Params": Conv2dGw8Params}
CONFIG_CLASSES = {
    "Conv2dGw8Config": Conv2dGw8Config,
    "Conv2dGw8WgradConfig": Conv2dGw8WgradConfig,
}

device_arch_table = {
    "nvidia_a100-pcie-40gb": "sm_80",
    "nvidia_a100-sxm4-80gb" : "sm_80",
    "nvidia_geforce_rtx_3090": "sm_86",
    "nvidia_geforce_rtx_4090": "sm_89",
}

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None


def train_performance_models(data_dir: str):
    """Train performance models for all spio kernels found in the given data directory."""
    device = get_device_name_from_data_dir(data_dir)
    try:
        arch = device_arch_table[device]
    except KeyError as e:
        raise ValueError(f"Unknown device: {device}. Add to device_arch_table") from e

    df = read_ssv_files_to_dataframe(data_dir, device)

    # Extract the "Kernel" column and split to get the root names
    df["RootName"] = df["Kernel"].str.split("__").str[0]

    # Group the DataFrame by the root names
    grouped = df.groupby("RootName")

    # Create a dictionary to store each group as a separate DataFrame
    dataframes = {
        root_name: group.drop(columns=["Kernel", "RootName"])
        for root_name, group in grouped
    }

    kernels = dataframes.keys()

    print(f"Training model for kernels {list(kernels)} on {device} ({arch})")
    print(f"Total number of samples {len(df)}")

    for kernel, df in dataframes.items():
        print(f"Kernel {kernel} has {len(df)} total samples.")
        df = average_redundant_params_and_configs(df)

        print()
        print(f"Kernel {kernel} has {len(df)} unique samples.")

        reflection = get_kernel_reflection(kernel)

        # Separate the features and target.
        X = df[["Params", "Config"]]
        y = df["CUDA_time_avg_ms"]

        X = import_dataclass_column(X, "Params", PARAMS_CLASSES)

        drop_ignored_params(X, reflection.ignore_params)

        X = import_dataclass_column(X, "Config", CONFIG_CLASSES)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Convert data to DMatrix format (optimized for XGBoost)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Set XGBoost parameters
        params = {
            "objective": "reg:squarederror",  # Regression task
            "max_depth": 6,  # Maximum depth of the trees
            "eta": 0.1,  # Learning rate
            "subsample": 0.8,  # Fraction of data to be used for each tree
            "colsample_bytree": 0.8,  # Fraction of features to be used for each tree
            "seed": 42,  # For reproducibility
        }

        num_rounds = 100  # Number of boosting rounds

        # Grid search for best training parameters.
        gscv = GridSearchCV(
            xgb.XGBRegressor(),
            {
                "max_depth": [3, 6, 9],
                "eta": [0.1, 0.3, 0.5],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            verbose=1,
            cv=5,
        )
        the_fit = gscv.fit(X_train, y_train)
        print(the_fit.best_params_)
        the_best_params = dict(params, **the_fit.best_params_)

        # Train the model
        bst = xgb.train(the_best_params, dtrain, num_rounds)

        # Evaluate the model on the test set
        y_pred = bst.predict(dtest)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"Root Mean Squared Error: {rmse:.3f} milliseconds")

        print(f"The mean of the target is {y_test.mean():.3f} +/- {y_test.std():.3f}")

        output_file_name = get_device_performance_model_file_name(kernel, device, arch)

        if "." not in output_file_name:
            output_file_name += PERFORMANCE_MODEL_EXTENSION

        print(f"Saving model to {output_file_name}")

        bst.save_model(output_file_name)


def regression_analysis(old_data_dir: str, new_data_dir: str):
    """Perform regression analysis between old and new benchmark results.

    Reports speedup based on average CUDA execution time for matching spio
    kernels, parameters, and configurations.
    """
    if not os.path.exists(old_data_dir):
        raise ValueError(f"Old data directory does not exist: {old_data_dir}")
    if not os.path.exists(new_data_dir):
        raise ValueError(f"New data directory does not exist: {new_data_dir}")
    old_device = get_device_name_from_bench_dir(old_data_dir)
    new_device = get_device_name_from_bench_dir(new_data_dir)
    assert (
        old_device == new_device
    ), f"Devices do not match: {old_device} != {new_device}"
    device = new_device
    assert device in device_arch_table, f"Unknown device: {device}"

    old_df = read_ssv_files_to_dataframe(old_data_dir, device)
    new_df = read_ssv_files_to_dataframe(new_data_dir, device)

    numeric_columns = ["CUDA_time_avg_ms"]

    # Group by Kernel, Params, and Config and compute the mean of numeric columns
    old_df = old_df.groupby(["Kernel", "Params", "Config"], as_index=False)[
        numeric_columns
    ].mean()
    new_df = new_df.groupby(["Kernel", "Params", "Config"], as_index=False)[
        numeric_columns
    ].mean()

    # Merge old and new dataframes on Kernel, Params, and Config
    merged_df = pd.merge(
        old_df, new_df, on=["Kernel", "Params", "Config"], suffixes=("_old", "_new")
    )

    # Compute speedup
    merged_df["speedup"] = (
        merged_df["CUDA_time_avg_ms_old"] / merged_df["CUDA_time_avg_ms_new"]
    )

    # Sort the DataFrame by Kernel, Params, and CUDA_time_avg_ms_new
    merged_df = merged_df.sort_values(by=["Kernel", "Params", "CUDA_time_avg_ms_new"])

    # Print the DataFrame with blank lines between groups
    previous_kernel = None
    previous_params = None
    rows = []
    for _, row in merged_df.iterrows():
        current_kernel = row["Kernel"]
        current_params = row["Params"]
        if previous_kernel is not None and (
            current_kernel != previous_kernel or current_params != previous_params
        ):
            rows.append([""] * len(row))

        rows.append(
            [
                row["Kernel"],
                row["Params"],
                row["Config"],
                f"{row['CUDA_time_avg_ms_old']:.3f}",
                f"{row['CUDA_time_avg_ms_new']:.3f}",
                f"{row['speedup']:.3f}",
            ]
        )
        previous_kernel = current_kernel
        previous_params = current_params
    print(
        tabulate(
            rows,
            headers=[
                "Kernel",
                "Params",
                "Config",
                "CUDA_time_avg_ms_old",
                "CUDA_time_avg_ms_new",
                "speedup",
            ],
            tablefmt="plain",
        )
    )


def main():
    """Main function to train the performance model or perform regression analysis."""
    parser = argparse.ArgumentParser(
        description="Train a performance model or perform regression analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train a performance model for a given kernel."
    )
    train_parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        help="Path to the directory containing samples for a device.",
    )

    regression_parser = subparsers.add_parser(
        "regression", help="Perform regression analysis between old and new data."
    )
    regression_parser.add_argument(
        "--old-data-dir",
        required=True,
        type=str,
        help="Path to the directory containing old samples.",
    )
    regression_parser.add_argument(
        "--new-data-dir",
        required=True,
        type=str,
        help="Path to the directory containing new samples.",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_performance_models(args.data_dir)
    elif args.command == "regression":
        regression_analysis(args.old_data_dir, args.new_data_dir)


def read_ssv_files_to_dataframe(directory: str, device: str) -> pd.DataFrame:
    """Return a pandas dataframe containing the data from all .ssv files in the directory tree."""
    ssv_files = glob.glob(os.path.join(directory, "**", "*.ssv"), recursive=True)
    for ssv_file in ssv_files:
        ssv_file_device = get_device_name_from_ssv_file_name(ssv_file)
        assert (
            ssv_file_device == device
        ), f"Device mismatch: {ssv_file_device} != {device}"
    dataframes = [pd.read_csv(file, delimiter=";") for file in ssv_files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def get_first_record(datafile: str):
    """Return the first record from the SSV file."""
    field_names = []
    with open(datafile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = line.split(";")
                # Skip header
                if tokens[0] == "Kernel":
                    field_names = tokens
                    continue
                record = {}
                for field_name, token in zip(field_names, tokens):
                    record[field_name] = token
                return record
    return None


def get_device_name_from_ssv_file_name(ssv_file_name: str) -> str:
    """Extract the device name from the SSV file path."""
    path = Path(ssv_file_name)
    parent = path.parent
    parent_dir_name = parent.name
    return get_device_name_from_data_dir(parent_dir_name)


def get_device_name_from_bench_dir(dir_name: str) -> str:
    """Decode the name of a results directory from a layer benchmark.

    Returns the components of the name as a dictionary.
    """
    child = Path(dir_name).name
    parts = child.split("__")
    if parts[0] != "bench":
        raise ValueError(
            f"Invalid benchmark directory name: {dir_name}: Expected 'bench__<device_name>..."
        )
    device_name = parts[1]
    return device_name


def get_device_name_from_data_dir(dir_name: str) -> str:
    """Decode the name of a results directory from a model benchmark.

    Returns the components of the name as a dictionary.
    """
    child = Path(dir_name).name
    parts = child.split("___")
    if parts[0] != "modelbench":
        parts = child.split("__")
        if parts[0] != "bench":
            raise ValueError(
                f"Invalid model benchmark directory name: {dir_name}: "
                "Expected 'modelbench__<device_name> or bench_<device_name>..."
            )
    device_name = parts[1]
    return device_name


def drop_ignored_params(df, ignore_params=None):
    """Drop any ignored params from the DataFrame.

    The kernel's ignored params are retrieved from the kernel reflection's
    "ignore_params" attribute.
    """
    if ignore_params is not None:
        ignore_fields = [f"Params_{field}" for field in ignore_params]
        for field in ignore_fields:
            df.drop(field, axis=1, inplace=True)
    return df


def average_redundant_params_and_configs(df):
    """Average the CUDA_time_avg_ms for redundant Params x Config."""

    # Convert Params and Config to strings or tuples
    df["Params_str"] = df["Params"].astype(str)
    df["Config_str"] = df["Config"].astype(str)

    # Group the DataFrame
    grouped = df.groupby(["Params_str", "Config_str"])

    # Compute the mean of CUDA_time_avg_ms
    mean_results = grouped["CUDA_time_avg_ms"].mean().reset_index()

    # (Optional) Merge with other fields
    other_fields = grouped.first().reset_index()
    other_fields = other_fields.drop(columns=["CUDA_time_avg_ms"])

    averaged_df = pd.merge(mean_results, other_fields, on=["Params_str", "Config_str"])
    averaged_df.drop(["Params_str", "Config_str"], axis=1, inplace=True)

    return averaged_df


def _dataclass_to_series(dataclass_obj):
    data = {}
    for key, value in dataclass_obj.__dict__.items():
        if isinstance(value, tuple):
            for i, elem in enumerate(value):
                data[f"{key}_{i}"] = elem
        else:
            data[key] = value
    return pd.Series(data)


def import_dataclass_column(df, col, dataclasses):
    """Parse the dataclass column and expand it into separate columns."""
    parse_these_dataclasses = partial(parse_dataclass, dataclasses=dataclasses)
    df[col] = df[col].apply(parse_these_dataclasses)
    attributes_df = df[col].apply(_dataclass_to_series)

    # Prefix the attribute column names to avoid conflicts
    attributes_df = attributes_df.add_prefix(f"{col}_")

    # Join the new attributes DataFrame with the original DataFrame
    df = pd.concat([df, attributes_df], axis=1)

    # Drop the original dataclass column if no longer needed
    df.drop(columns=[col], inplace=True)
    return df


if __name__ == "__main__":
    main()
