"""Create the kernel factory for the MLP kernel."""

from dataclasses import dataclass
from typing import Generator, List

import torch

from .. import generators as gen
from ..util import next_relative_prime
from ..cuda.driver import FunctionAttributes, DeviceAttributes


from .launch_params import LaunchParams
from .mlp_params import MlpParams
from .mlp_stats import MlpStats
from .kernel_factory import KernelFactory
from .kernel import get_full_kernel_name, KernelSpec


NUM_SMEM_BANKS_32 = 32
NUM_SMEM_BANKS_128 = NUM_SMEM_BANKS_32 // 4

UNIT = 2
PING_PONG = 2

ACTIVATIONS = ["relu", "gelu", "silu", None]
KERNEL_NAME = "mlp_tiny_c"


@dataclass(frozen=True)
class MlpConfig:
    """Tile configuration for the MLP kernel."""

    warp_x16: int = 2
    r16_chunk: int = 1
    warps: int = 16

    @property
    def warp_x(self) -> int:
        """Return the x-sisze of a warp."""
        return self.warp_x16 * 16

    @property
    def block_x(self) -> int:
        """Return the x-size of a block."""
        return self.warp_x * self.warps

    @property
    def r_chunk(self):
        """Return the r-size of a chunk."""
        return self.r16_chunk * 16

    @property
    def threads(self):
        """Return the number of threads in a block."""
        return self.warps * 32


def _get_configs(params: MlpParams, **_kwargs) -> Generator[MlpConfig, None, None]:
    """Generate configurations for the MLP kernel."""
    k = params.k
    if k <= 128:
        yield MlpConfig(
            warp_x16=1,
            r16_chunk=1,
            warps=16,
        )
        return
    elif k <= 64:
        yield MlpConfig(warp_x16=2, r16_chunk=1, warps=16)
    else:
        raise ValueError(f"Invalid k: {k}")


def _get_kernel_spec(params: MlpParams, config: MlpConfig, device_attr: DeviceAttributes ) -> KernelSpec:
    """Return the generator specs, grid and block for the MLP kernel."""
    _validate_params(params)

    blocks = device_attr.multiprocessor_count

    smem_size = device_attr.max_shared_memory_per_block_optin
    smem_carveout = 100

    c8 = params.c // 8
    k8 = params.k // 8
    c16 = params.c // 16
    r16 = params.r // 16
    k16 = params.k // 16

    input_tensor = gen.Tensor(
        "Input", gen.dtype.uint4, gen.Dims(x=params.x, c8=c8), constant=True
    )
    output_tensor = gen.Tensor("Output", gen.dtype.uint4, gen.Dims(x=params.x, k8=k8))
    exp_weights_tensor = gen.Tensor(
        "ExpWeight", gen.dtype.uint4, gen.Dims(c16=c16, r=params.r, c8=2), constant=True
    )
    prj_weights_tensor = gen.Tensor(
        "PrjWeight", gen.dtype.uint4, gen.Dims(r16=r16, k=params.k, r8=2), constant=True
    )

    input_smem_tensor_bytes = 32 * 1024
    input_smem_tensor_size = input_smem_tensor_bytes // 16
    input_smem_buffer_size = c8 * config.block_x
    num_input_smem_buffers = input_smem_tensor_size // input_smem_buffer_size
    smem_input_x_stride = next_relative_prime(c8, NUM_SMEM_BANKS_128)

    smem_input_tensor = gen.Tensor(
        "SmemInput",
        gen.dtype.uint4,
        gen.Dims(buffer=num_input_smem_buffers, x=config.warp_x, c8=c8),
        gen.Strides(x=smem_input_x_stride),
    )

    smem_exp_weights_tensor = gen.Tensor(
        "SmemExpWeight",
        gen.dtype.uint4,
        gen.Dims(c16=c16, r16=r16, checkers=gen.CheckerboardIndex(r=16, c8=2)),
    )

    smem_prj_weights_tensor = gen.Tensor(
        "SmemPrjWeight",
        gen.dtype.uint4,
        gen.Dims(r16=r16, k16=k16, checkers=gen.CheckerboardIndex(k=16, r8=2)),
    )

    specs = [
        gen.Macro({"SPIO_MLP_KERNEL": get_full_kernel_name(KERNEL_NAME, params)}),
        gen.Fold("block_x", "x", config.block_x),
        gen.ParamsSpec("Params", dict(blocks=blocks)),
        gen.Index("WarpIdx", {"warp": config.warps, "lane": 32}),
        input_tensor,
        output_tensor,
        exp_weights_tensor,
        prj_weights_tensor,
        smem_input_tensor,
        smem_exp_weights_tensor,
        smem_prj_weights_tensor,
    ]

    launch_params = LaunchParams(
        grid=blocks, block=config.threads, shared_mem_bytes=smem_size
    )

    function_attributes = FunctionAttributes(
        max_dynamic_shared_memory_size=smem_size,
        preferred_shared_memory_carveout=smem_carveout,
    )

    return KernelSpec(
        gen_specs=specs,
        launch_params=launch_params,
        function_attributes=function_attributes,
    )


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
    kernel_spec=_get_kernel_spec,
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
