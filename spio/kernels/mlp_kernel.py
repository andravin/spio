"""Create the kernel factory for the MLP kernel."""

from dataclasses import dataclass
from typing import Generator, Tuple, List

import torch

from ..generators import (
    MacroSpec,
    IndexSpec,
    ParamsSpec,
    TensorSpec,
    GenSpecs,
)
from ..util import divup, next_relative_prime
from .launch_params import LaunchParams
from .mlp_params import MlpParams
from .mlp_stats import MlpStats
from .kernel_factory import KernelFactory
from .kernel import get_full_kernel_name


# TODO Get device limits from the device attributes.
MAX_SMEM = 99 * 1024

UNIT = 2
PING_PONG = 2

ACTIVATIONS = ["relu", "gelu", "silu", None]
KERNEL_NAME = "mlp_small_c"


@dataclass(frozen=True)
class MlpConfig:
    """Tile configuration for the MLP kernel."""

    warp_x16: int
    r16_chunk: int
    warps_x: int

    @property
    def warp_x(self) -> int:
        """Return the x-sisze of a warp."""
        return self.warp_x16 * 16

    @property
    def block_x(self) -> int:
        """Return the x-size of a block."""
        return self.warp_x * self.warps_x


def _get_configs(params: MlpParams, **_kwargs) -> Generator[MlpConfig, None, None]:
    """Generate configurations for the MLP kernel."""
    k = params.k
    if k <= 256:
        yield MlpConfig(
            warp_x16=1,
            r16_chunk=1,
            warps_x=8,
        )
        if params.r % 32 == 0:
            yield MlpConfig(
                warp_x16=1,
                r16_chunk=2,
                warps_x=8,
            )
        return
    raise ValueError(f"Invalid k: {k}")


def _get_specs(
    params: MlpParams, config: MlpConfig = None
) -> Tuple[List[GenSpecs], LaunchParams]:
    _validate_params(params)

    c8 = divup(params.c, 8)
    r8 = divup(params.r, 8)
    k8 = divup(params.k, 8)
    c16 = divup(params.c, 16)
    r16 = divup(params.r, 16)
    k16 = divup(params.k, 16)
    num_r_chunks = divup(params.r, config.r16_chunk * 16)
    warp_x8 = config.warp_x16 * 2
    block_x8 = config.block_x // 8

    full_kernel_name = get_full_kernel_name(KERNEL_NAME, params)

    blocks = divup(params.x, config.block_x)
    warps = config.warps_x
    threads = warps * 32

    # When storing 8-element float16 vectors, smem effectively has 8 banks.
    num_smem_banks = 8
    smem_input_x_stride = next_relative_prime(c8, num_smem_banks)

    # Store the input tensor tile in shared memory using a checkerboard layout.
    smem_input_tensor = TensorSpec(
        "SmemInput",
        "uint4",
        {
            "warp_x": config.warps_x,
            "x": config.warp_x,
            "c8": c8,
        },
        strides={"x": smem_input_x_stride},
    )

    # Store the expansion weight tensor tiles in shared memory.
    # Pad the r-dimension stride to avoid bank conflicts.
    smem_exp_weight_tensor = TensorSpec(
        "SmemExpWeight",
        "uint4",
        {"ping_pong": 2, "r16": config.r16_chunk, "c16": c16, "checkerboard": 32},
    )

    # Store the projection weight tensor tiles in shared memory.
    # Use a checkerboard layout.
    smem_prj_weight_tensor = TensorSpec(
        "SmemPrjWeight",
        "uint4",
        {"ping_pong": 2, "k16": k16, "r16": config.r16_chunk, "checkerboard": 32},
    )

    assert warps % config.r16_chunk == 0, "Warps must be a multiple of r16_chunk."
    warps_c16 = warps // config.r16_chunk
    warps_k16 = warps // config.r16_chunk

    specs = [
        MacroSpec({"SPIO_MLP_KERNEL": full_kernel_name}),
        #
        # Block parameters.
        ParamsSpec("Block", {"x": config.block_x, "warps": warps, "threads": threads}),
        #
        # Constants.
        ParamsSpec(
            "Params",
            {
                "r8": r8,
                "warp_x16": config.warp_x16,
                "r16_chunk": config.r16_chunk,
                "c16": c16,
                "k16": k16,
                "num_r_chunks": num_r_chunks,
            },
        ),
        #
        # Warp / lane indices.
        IndexSpec("WarpIdx", {"warp": warps, "lane": 32}),
        #
        # Input, expansion weight, projection weight, and output tensors.
        TensorSpec("Input", "const uint4", {"x": params.x, "c8": c8}),
        TensorSpec("ExpWeight", "const uint4", {"c16": c16, "r": params.r, "c8": 2}),
        TensorSpec("PrjWeight", "const uint4", {"r16": r16, "k": params.k, "r8": 2}),
        TensorSpec("Output", "uint4", {"x": params.x, "k8": k8}),
        #
        # Shared memory tensors.
        smem_input_tensor,
        smem_exp_weight_tensor,
        smem_prj_weight_tensor,
        TensorSpec(
            "SmemOutputStore",
            "__half2",
            {"x8": block_x8, "k8": k8, "lane": 32},
            strides={"k8": 36}, # 32 half2 + 4 half2 padding
        ),
        TensorSpec(
            "SmemOutputLoad",
            "uint4",
            {"warp": config.warps_x, "x8": warp_x8, "k8": k8, "xm8": 8},
            strides={"k8": 9}, # 8 uint4 + 1 uint4 padding
        ),
        IndexSpec("SmemOutputLoadIdx", {"x8": warp_x8, "xm8": 8, "k8": k8}),
        #
        # Matrix fragments.
        TensorSpec("In", "spio::MMA_M16_K16_F16_A", {"c16": c16, "x16": config.warp_x16}),
        TensorSpec(
            "Hidden",
            "spio::MMA_M16_N16_F32_C",
            {"r16": config.r16_chunk, "x16": config.warp_x16},
        ),
        TensorSpec(
            "HiddenAct",
            "spio::MMA_M16_K16_F16_A",
            {"r16": config.r16_chunk, "x16": config.warp_x16},
        ),
        TensorSpec("Exp", "spio::MMA_N16_K16_F16_B", {"r16": config.r16_chunk, "c16": c16}),
        TensorSpec("Prj", "spio::MMA_N16_K16_F16_B", {"k16": k16, "r16": config.r16_chunk}),
        TensorSpec("Out", "spio::MMA_M16_N16_F32_C", {"k16": k16, "x16": config.warp_x16}),
        IndexSpec(
            "ExpWeightLoadIdx", {"warp_c16": warps_c16, "warp_r16": config.r16_chunk}
        ),
        IndexSpec(
            "PrjWeightLoadIdx", {"warp_k16": warps_k16, "warp_r16": config.r16_chunk}
        ),
    ]

    # Allocate shared memory.
    smem_buf1_size = smem_input_tensor.num_bytes
    smem_buf2_size = smem_exp_weight_tensor.num_bytes + smem_prj_weight_tensor.num_bytes
    smem_size = max(smem_buf1_size, smem_buf2_size)
    assert (
        smem_size <= MAX_SMEM
    ), f"Shared memory exceeds limit: {smem_size} > {MAX_SMEM}."

    launch_params = LaunchParams(grid=blocks, block=threads, shared_mem_bytes=smem_size)

    return specs, launch_params


def _check_args(args: List[torch.Tensor]) -> None:
    """Check the arguments for the MLP kernel."""
    assert args[0].dim() == 2, "Output tensor must be 2D."
    assert args[1].dim() == 2, "Input tensor must be 2D."
    assert args[2].dim() == 3, "Expansion weight tensor must be 3D."
    assert args[2].shape[2] == 16, "Expansion weight tensor must be 16c last."
    assert args[3].dim() == 1, "Expansion bias tensor must be 1D."
    assert args[4].dim() == 3, "Projection weight tensor must be 3D."
    assert args[4].shape[2] == 16, "Projection weight tensor must be 16r last."
    assert args[5].dim() == 1, "Projection bias tensor must be 1D."
    assert len(args) == 6, "Invalid number of arguments."


mlp_kernel_factory = KernelFactory(
    MlpParams,
    MlpConfig,
    MlpStats,
    kernel_name=KERNEL_NAME,
    configs=_get_configs,
    specs=_get_specs,
    kernel_source_file="mlp.cu",
    src_module="spio.src",
    includes_module="spio.include",
    args_checker=_check_args,
)


def _validate_params(params: MlpParams) -> None:
    assert params.c <= 256, "Input channels must be less than or equal to 256."
    assert params.k <= 256, "Output channels must be less than or equal to 256."
    assert params.c % 16 == 0, "Input channels must be a multiple of 16."
    assert params.r % 16 == 0, "Hidden channels must be a multiple of 16."
    assert params.k % 16 == 0, "Output channels must be a multiple of 16."
    assert params.x > 0, "Input samples must be positive."
    assert params.c > 0, "Input channels must be positive."
    assert params.r > 0, "Hidden channels must be positive."
    assert params.k > 0, "Output channels must be positive."
    assert params.activation in ACTIVATIONS, f"Invalid activation: {params.activation}"
    assert params.bias in [True, False], "Bias must be a boolean."
