"""Script for plotting kernel benchmark results."""

import os
import re
from argparse import ArgumentParser
from pathlib import Path
from dataclasses import replace
from functools import partial

import pandas as pd
import matplotlib.pyplot as plt

from spio.src_tests import preprocess_data_file
from spio.reflection import get_kernel_reflection


MAX_BANDWIDTH_GB_S = 3000


def main():
    """Main function to plot the benchmark results."""
    parser = ArgumentParser()
    parser.add_argument(
        "spio_data_dir", type=str, help="Path to the Spio benchmark data directory"
    )
    parser.add_argument(
        "--torch-data-dir",
        type=str,
        help="Path to the PyTorch benchmark data directory",
    )
    parser.add_argument("--max-bandwidth-gb-s", type=float, default=MAX_BANDWIDTH_GB_S)
    args = parser.parse_args()

    if not Path(args.spio_data_dir).is_dir():
        raise ValueError(f"Invalid data directory: {args.spio_data_dir}")

    if args.torch_data_dir is not None and not Path(args.torch_data_dir).is_dir():
        raise ValueError(f"Invalid data directory: {args.torch_data_dir}")

    # Collect data from all files
    df = collect_data_from_dir(args.spio_data_dir)

    # Concatenate all DataFrames into a single DataFrame
    df = df.sort_values(by="batch_size")

    fprop_kernel_name = "spio_conv2d_gw8_fprop"
    dgrad_kernel_name = "spio_conv2d_gw8_dgrad"
    wgrad_kernel_name = "spio_conv2d_gw8_wgrad"
    spio_kernel_labels = ["Spio Fprop", "Spio Dgrad", "Spio Wgrad"]

    # Extract the "Kernel" column and split to get the root names
    df["RootName"] = df["Name"].str.split("__").str[0]

    # Group the DataFrame by the root names
    grouped = df.groupby("RootName")

    # Create a dictionary to store each group as a separate DataFrame
    dataframes = {
        root_name: group.drop(columns=["Name", "RootName"])
        for root_name, group in grouped
    }

    df_fprop = dataframes[fprop_kernel_name]
    df_dgrad = dataframes[dgrad_kernel_name]
    df_wgrad = dataframes[wgrad_kernel_name]

    dirname_params = extract_parameters_from_dirname(args.spio_data_dir)
    params = conv2d_gw8_kernel_params_from_dirname_params(dirname_params)

    fprop_reflection = get_kernel_reflection(fprop_kernel_name)
    dgrad_reflection = get_kernel_reflection(dgrad_kernel_name)
    wgrad_reflection = get_kernel_reflection(wgrad_kernel_name)

    add_eff_bandwidth_gb_s(df_fprop, params, fprop_reflection.stats, "output")
    add_eff_bandwidth_gb_s(df_dgrad, params, dgrad_reflection.stats, "grad_input")
    add_eff_bandwidth_gb_s(df_wgrad, params, wgrad_reflection.stats, "grad_weight")

    plot_eff_bw(
        df_fprop,
        df_dgrad,
        df_wgrad,
        spio_kernel_labels,
        linestyle="-",
    )
    block_name = dirname_params["block_name"]

    if args.torch_data_dir is not None:
        torch_dirname_params = extract_parameters_from_dirname(args.torch_data_dir)
        torch_params = conv2d_gw8_kernel_params_from_dirname_params(
            torch_dirname_params
        )
        torch_params_are_depthwise = torch_params.group_width == 1

        assert (
            dirname_params["block_name"] == torch_dirname_params["block_name"]
        ), f"Block name mismatch: {dirname_params['block_name']} != {torch_dirname_params['block_name']}"
        assert (
            dirname_params["device_name"] == torch_dirname_params["device_name"]
        ), f"Device name mismatch: {dirname_params['device_name']} != {torch_dirname_params['device_name']}"

        df = collect_data_from_dir(args.torch_data_dir)
        df = df.sort_values(by="batch_size")
        kernel_iters = compute_num_iters_from_dirname_params(torch_dirname_params)
        fprop_df, dgrad_df, wgrad_df = find_torch_grouped_conv_kernels(
            df, kernel_iters, depthwise=torch_params_are_depthwise
        )
        torch_kernel_labels = [
            "PyTorch" + kernel_name.removeprefix("Spio")
            for kernel_name in spio_kernel_labels
        ]
        if torch_params_are_depthwise:
            torch_kernel_labels = [
                label + " (Depthwise)" for label in torch_kernel_labels
            ]

        add_eff_bandwidth_gb_s(fprop_df, torch_params, fprop_reflection.stats, "output")
        add_eff_bandwidth_gb_s(
            dgrad_df, torch_params, dgrad_reflection.stats, "grad_input"
        )
        add_eff_bandwidth_gb_s(
            wgrad_df, torch_params, wgrad_reflection.stats, "grad_weight"
        )

        plot_eff_bw(
            fprop_df,
            dgrad_df,
            wgrad_df,
            torch_kernel_labels,
            linestyle="--",
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Effective Bandwidth (GB/s)")
    plt.ylim(0, args.max_bandwidth_gb_s)
    device_name = dirname_params["device_name"]
    plt.title("Grouped Convolution Performance")
    plt.suptitle(
        f"{params.C} channels, {params.H}x{params.W} input, {params.R}x{params.S} kernel, group width {params.group_width}, {device_name}",
        fontsize=10,
    )
    plt.legend()

    fig_file_name = f"batch_size_vs_eff_bandwidth__{device_name}__{block_name}_{params.C}c_{params.R}r_{params.S}s_{params.group_width}gw"
    if args.torch_data_dir is not None and torch_params_are_depthwise:
        fig_file_name += "__torch_depthwise"
    fig_file_name += ".png"

    plt.savefig(fig_file_name)
    print(f"Saved figure to {fig_file_name}")

    plt.clf()
    plot_latency_microseconds(
        df_fprop, df_dgrad, df_wgrad, spio_kernel_labels, linestyle="-"
    )
    if args.torch_data_dir is not None:
        plot_latency_microseconds(
            fprop_df, dgrad_df, wgrad_df, torch_kernel_labels, linestyle="--"
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Latency (microseconds)")
    plt.title("Latency vs Batch Size")
    plt.suptitle(
        f"{params.C} channels, {params.H}x{params.W} input, {params.R}x{params.S} kernel, group width {params.group_width}, {device_name}",
        fontsize=10,
    )
    plt.legend()

    fig_file_name = f"batch_size_vs_latency__{device_name}__{block_name}_{params.C}c_{params.R}r_{params.S}s_{params.group_width}gw"
    if args.torch_data_dir is not None and torch_params_are_depthwise:
        fig_file_name += "__torch_depthwise"
    fig_file_name += ".png"
    plt.savefig(fig_file_name)
    print(f"Saved figure to {fig_file_name}")


def find_torch_grouped_conv_kernels(df, kernel_iters, depthwise=False):
    """Find and group the PyTorch convolution kernels in the DataFrame."""
    grouped_fprop_kernels = [
        "sm86_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize64x64x64_stage3_warpsize1x4x1_g8_tensor16x8x16_t1r3s3_execute_kernel__5x_cudnn",
        "sm86_xmma_fprop_implicit_gemm_indexed_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize64x64x64_stage3_warpsize1x4x1_g8_tensor16x8x16_execute_kernel__5x_cudnn",
    ]

    grouped_dgrad_kernels = [
        "sm80_xmma_dgrad_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize64x64x64_stage4_warpsize1x4x1_g8_tensor16x8x16_execute_kernel__5x_cudnn",
        "sm80_xmma_dgrad_implicit_gemm_indexed_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize64x64x64_stage4_warpsize1x4x1_g8_tensor16x8x16_execute_kernel__5x_cudnn",
    ]

    grouped_wgrad_kernels = [
        "sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize64x64x64_stage4_warpsize1x1x4_g8_tensor16x8x16_execute_kernel__5x_cudnn",
    ]

    dw_fprop_kernels = [
        "void conv2d_c1_k1_nhwc_kernel_specialized<__half, __half, __half, float, float, 5, 1, 8, 3, true>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, C1K1NhwcParams, int)",
        "void conv2d_c1_k1_nhwc_kernel_specialized<__half, __half, __half, float, float, 3, 1, 8, 3, true>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, C1K1NhwcParams, int)",
        "void conv2d_c1_k1_nhwc_kernel_specialized<__half, __half, __half, float, float, 7, 1, 8, 3, true>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, C1K1NhwcParams, int)",
    ]

    dw_dgrad_kernels = [
        "sm80_xmma_depthwise_convolution_dgrad_tiling_f16f16_f32f32_f32_nhwckrsc_nhwc_tilesize1x8x8_1x5x5_1x1x1_1x1x1_stage2_warpsize4x1x1_g64_betaFalse_execute_kernel__5x_cudnn",
        "sm80_xmma_depthwise_convolution_dgrad_tiling_f16f16_f32f32_f32_nhwckrsc_nhwc_tilesize1x8x8_1x5x5_1x1x1_1x1x1_stage2_warpsize4x1x1_g64_betaFalse_execute_helper_kernel__5x_cudnn",
        "void dgrad2d_c1_k1_nhwc_kernel_specialized_window<__half, float, float, 5, 1, true>(float, __half const*, __half const*, float, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)",
        "void dgrad2d_c1_k1_nhwc_kernel_specialized_window<__half, float, float, 3, 1, true>(float, __half const*, __half const*, float, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)",
        "void dgrad2d_c1_k1_nhwc_kernel_specialized_window<__half, float, float, 7, 1, true>(float, __half const*, __half const*, float, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)",
    ]

    dw_wgrad_kernels = [
        "sm80_xmma_depthwise_convolution_wgrad_tiling_f16f16_f32f32_f32_nhwckrsc_nhwc_tilesize1x8x8_1x5x5_1x1x1_1x1x1_stage2_warpsize4x1x1_g32_betaTrue_execute_kernel__5x_cudnn",
        "sm80_xmma_depthwise_convolution_wgrad_tiling_f16f16_f32f32_f32_nhwckrsc_nhwc_tilesize1x8x8_1x5x5_1x1x1_1x1x1_stage2_warpsize4x1x1_g32_betaTrue_post_execute_kernel__5x_cudnn",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_kernel<__half, float, 1, 5, 5, 1, 1, 1, 1, true>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnConvolutionStruct, cudnnFilterStruct, float*)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_reduction_kernel<__half, float, 1, 5, 5>(cudnnTensorStruct, cudnnConvolutionStruct, __half*, float const*, float, float)",
        "sm80_xmma_depthwise_convolution_wgrad_tiling_f16f16_f32f32_f32_nhwckrsc_nhwc_tilesize1x4x4_1x5x5_1x1x1_1x1x1_stage4_warpsize4x1x1_g32_betaTrue_execute_kernel__5x_cudnn",
        "sm80_xmma_depthwise_convolution_wgrad_tiling_f16f16_f32f32_f32_nhwckrsc_nhwc_tilesize1x4x4_1x5x5_1x1x1_1x1x1_stage4_warpsize4x1x1_g32_betaTrue_post_execute_kernel__5x_cudnn",
        "sm80_xmma_depthwise_convolution_wgrad_tiling_f16f16_f32f32_f32_nhwckrsc_nhwc_tilesize1x8x8_1x3x3_1x1x1_1x1x1_stage4_warpsize4x1x1_g32_betaTrue_execute_kernel__5x_cudnn",
        "sm80_xmma_depthwise_convolution_wgrad_tiling_f16f16_f32f32_f32_nhwckrsc_nhwc_tilesize1x8x8_1x3x3_1x1x1_1x1x1_stage4_warpsize4x1x1_g32_betaTrue_post_execute_kernel__5x_cudnn",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_kernel<__half, float, 1, 3, 3, 1, 1, 1, 1, true>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnConvolutionStruct, cudnnFilterStruct, float*)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_kernel<__half, float, 3, 3, 3, 1, 1, 1, 1, true>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnConvolutionStruct, cudnnFilterStruct, float*)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_kernel<__half, float, 8, 3, 3, 1, 1, 1, 1, true>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnConvolutionStruct, cudnnFilterStruct, float*)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_kernel<__half, float, 8, 7, 7, 1, 1, 1, 1, true>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnConvolutionStruct, cudnnFilterStruct, float*)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_kernel<__half, float, 7, 7, 7, 1, 1, 1, 1, true>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnConvolutionStruct, cudnnFilterStruct, float*)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_kernel<__half, float, 4, 7, 7, 1, 1, 1, 1, true>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnConvolutionStruct, cudnnFilterStruct, float*)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_kernel<__half, float, 3, 7, 7, 1, 1, 1, 1, true>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnConvolutionStruct, cudnnFilterStruct, float*)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_kernel<__half, float, 1, 7, 7, 1, 1, 1, 1, true>(cudnnTensorStruct, __half const*, cudnnTensorStruct, __half const*, cudnnConvolutionStruct, cudnnFilterStruct, float*)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_reduction_kernel<__half, float, 1, 3, 3>(cudnnTensorStruct, cudnnConvolutionStruct, __half*, float const*, float, float)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_reduction_kernel<__half, float, 3, 3, 3>(cudnnTensorStruct, cudnnConvolutionStruct, __half*, float const*, float, float)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_reduction_kernel<__half, float, 8, 3, 3>(cudnnTensorStruct, cudnnConvolutionStruct, __half*, float const*, float, float)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_reduction_kernel<__half, float, 8, 7, 7>(cudnnTensorStruct, cudnnConvolutionStruct, __half*, float const*, float, float)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_reduction_kernel<__half, float, 7, 7, 7>(cudnnTensorStruct, cudnnConvolutionStruct, __half*, float const*, float, float)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_reduction_kernel<__half, float, 4, 7, 7>(cudnnTensorStruct, cudnnConvolutionStruct, __half*, float const*, float, float)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_reduction_kernel<__half, float, 3, 7, 7>(cudnnTensorStruct, cudnnConvolutionStruct, __half*, float const*, float, float)",
        "void cudnn::cnn::wgrad2d_c1_k1_nhwc_reduction_kernel<__half, float, 1, 7, 7>(cudnnTensorStruct, cudnnConvolutionStruct, __half*, float const*, float, float)",
    ]

    if depthwise:
        fprop_kernels = dw_fprop_kernels
        dgrad_kernels = dw_dgrad_kernels
        wgrad_kernels = dw_wgrad_kernels
    else:
        fprop_kernels = grouped_fprop_kernels
        dgrad_kernels = grouped_dgrad_kernels
        wgrad_kernels = grouped_wgrad_kernels

    num_iters_field_name = "#_of_Call"
    mask = df[num_iters_field_name].apply(lambda x: x in kernel_iters)
    kernels_df = df[mask]

    batch_sizes = set(df["batch_size"])

    fprop_df = kernels_df[kernels_df["Name"].apply(lambda x: x in fprop_kernels)]
    dgrad_df = kernels_df[kernels_df["Name"].apply(lambda x: x in dgrad_kernels)]
    wgrad_df = kernels_df[kernels_df["Name"].apply(lambda x: x in wgrad_kernels)]

    fprop_batch_sizes = set(fprop_df["batch_size"])
    dgrad_batch_sizes = set(dgrad_df["batch_size"])
    wgrad_batch_sizes = set(wgrad_df["batch_size"])

    assert (
        batch_sizes == fprop_batch_sizes
    ), f"Batch sizes mismatch: {batch_sizes} != {fprop_batch_sizes}"
    assert (
        batch_sizes == dgrad_batch_sizes
    ), f"Batch sizes mismatch: {batch_sizes} != {dgrad_batch_sizes}"
    assert (
        batch_sizes == wgrad_batch_sizes
    ), f"Batch sizes mismatch: {batch_sizes} != {wgrad_batch_sizes}"

    fprop_df = fprop_df.groupby("batch_size").agg(
        {"CUDA_time_av": "sum", "batch_size": "first"}
    )
    dgrad_df = dgrad_df.groupby("batch_size").agg(
        {"CUDA_time_av": "sum", "batch_size": "first"}
    )
    wgrad_df = wgrad_df.groupby("batch_size").agg(
        {"CUDA_time_av": "sum", "batch_size": "first"}
    )

    return (fprop_df, dgrad_df, wgrad_df)


def compute_num_iters_from_dirname_params(dirname_params):
    """Compute the expected number of iterations based on the directory parameters."""
    depth = dirname_params["depth"]
    block_name = dirname_params["block_name"]
    num_iters = dirname_params["num_iters"]

    if block_name == "mbconv":
        expect_conv2d_gw8_fprop_iters = depth * num_iters
        expect_conv2d_gw8_dgrad_iters = expect_conv2d_gw8_fprop_iters
        expect_conv2d_gw8_wgrad_iters = expect_conv2d_gw8_fprop_iters
    elif block_name == "convfirst":
        expect_conv2d_gw8_fprop_iters = depth * num_iters
        expect_conv2d_gw8_dgrad_iters = (depth - 1) * num_iters
        expect_conv2d_gw8_wgrad_iters = expect_conv2d_gw8_fprop_iters
    else:
        raise ValueError(f"Unknown block name: {block_name}")
    return (
        expect_conv2d_gw8_fprop_iters,
        expect_conv2d_gw8_dgrad_iters,
        expect_conv2d_gw8_wgrad_iters,
    )


# Function to read and parse a data file
def read_data_file(file_path):
    """Read and parse the SSV profile results file."""
    processed_lines = preprocess_data_file(file_path)
    batch_size = int(re.search(r"bench_bs(\d+)\.dat", file_path).group(1))

    # Create a DataFrame from the processed lines
    from io import StringIO

    data_str = "\n".join(processed_lines)
    df = pd.read_csv(StringIO(data_str), sep=";")

    # Add batch size column
    df["batch_size"] = batch_size

    return df


def extract_parameters_from_dirname(dirname):
    """Extract parameters from the directory name."""
    pattern = re.compile(
        r"bench__(.+)__([^_]+)_([^_]+)_c(\d+)_ks(\d+)_er(\d+)_gw(\d+)_hw(\d+)_d(\d+)_extra(\d+)_iters(\d+)(_bs(\d+))?"
    )
    match = pattern.search(dirname)
    field_names = [
        "device_name",
        "backend_name",
        "block_name",
        "channels",
        "kernel_size",
        "expansion_ratio",
        "group_width",
        "height_width",
        "depth",
        "extra",
        "num_iters",
        "batch_size",
    ]
    string_fields = ["device_name", "backend_name", "block_name"]
    if not match:
        raise ValueError(f"Directory name does not match expected format: {dirname}")
    match_dict = {}
    for i, field_name in enumerate(field_names):
        if match.group(i + 1):
            val = match.group(i + 1)
            if field_name not in string_fields:
                val = int(val)
            match_dict[field_name] = val
    return match_dict


def collect_data_from_dir(dirname):
    """Collect data from all .dat files in the specified directory."""
    all_data = []
    for file_name in os.listdir(dirname):
        if file_name.endswith(".dat"):
            file_path = os.path.join(dirname, file_name)
            all_data.append(read_data_file(file_path))
    if len(all_data) == 0:
        raise ValueError(f"No data files found in directory: {dirname}")
    return pd.concat(all_data, ignore_index=True)


def add_eff_bandwidth_gb_s(df, params, stats_cls, output_names):
    """Add effective bandwidth column to the DataFrame."""
    df["eff_bw_gb_s"] = df.apply(
        partial(
            compute_eff_bandwidth_gb_s,
            params=params,
            stats_cls=stats_cls,
            output_names=output_names,
        ),
        axis=1,
    )


def compute_eff_bandwidth_gb_s(row, params=None, stats_cls=None, output_names=None):
    """Compute effective bandwidth in GB/s."""
    params = replace(params, N=row["batch_size"])
    stats = stats_cls(params=params, output_names=output_names)
    bytes_mb = stats.bytes / 1e6
    return bytes_mb / row["CUDA_time_av"]


def conv2d_gw8_kernel_params_from_dirname_params(dirname_params):
    """Get kernel parameters from the directory name parameters."""
    block_name = dirname_params["block_name"]

    if block_name == "convfirst":
        C = dirname_params["channels"]
    elif block_name == "mbconv":
        C = dirname_params["channels"] * dirname_params["expansion_ratio"]
    else:
        raise ValueError(f"Unknown block name: {block_name}")
    fprop_reflection = get_kernel_reflection("spio_conv2d_gw8_fprop")
    params_cls = fprop_reflection.params
    return params_cls(
        N=dirname_params.get("batch_size"),
        C=C,
        H=dirname_params["height_width"],
        W=dirname_params["height_width"],
        padding=dirname_params["kernel_size"] // 2,
        R=dirname_params["kernel_size"],
        S=dirname_params["kernel_size"],
        has_bias=False,
        group_width=dirname_params["group_width"],
    )


# Plot the data
def plot_eff_bw(df_fprop, df_dgrad, df_wgrad, kernel_names, linestyle="-", colors=None):
    """Plot effective bandwidth."""
    if colors is None:
        colors = [f"C{i}" for i in range(len(kernel_names))]
    for df, label, color in zip([df_fprop, df_dgrad, df_wgrad], kernel_names, colors):
        plt.plot(
            df["batch_size"],
            df["eff_bw_gb_s"],
            label=label,
            color=color,
            linestyle=linestyle,
        )


def plot_latency_microseconds(
    df_fprop, df_dgrad, df_wgrad, kernel_names, linestyle="-", colors=None
):
    """Plot latency in microseconds."""
    if colors is None:
        colors = [f"C{i}" for i in range(len(kernel_names))]
    for df, label, color in zip([df_fprop, df_dgrad, df_wgrad], kernel_names, colors):
        plt.plot(
            df["batch_size"],
            df["CUDA_time_av"] * 1000.0,
            label=label,
            color=color,
            linestyle=linestyle,
        )


if __name__ == "__main__":
    main()
