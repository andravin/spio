"""Create the kernel factory for the LayerNorm2d kernel."""

from dataclasses import dataclass
from typing import Generator, Tuple, List
from itertools import product

from .. import generators as gen
from ..util import divup
from .launch_params import LaunchParams
from .layernorm_2d_params import LayerNorm2dParams
from .layernorm_2d_stats import LayerNorm2dStats
from .kernel_factory import KernelFactory
from .kernel import get_full_kernel_name


@dataclass(frozen=True)
class LayerNorm2dConfig:
    """Tile configuration for the LayerNorm2d kernel."""

    warps_x: int
    warps_c: int
    warp_x: int
    reverse_x: bool

    @property
    def block_x(self) -> int:
        """Return the block size in the x dimension."""
        return self.warps_x * self.warp_x


def _smem_allocation(config: LayerNorm2dConfig, params: LayerNorm2dParams):
    input_unit = 2
    sum_unit = 4
    ping_pong = 2
    smem_input_buf_size = config.warps_x * params.c * input_unit * ping_pong
    smem_sum_buf = config.warps_c * config.warp_x * sum_unit
    smem_diff_sum_buf = smem_sum_buf
    return smem_input_buf_size + smem_sum_buf + smem_diff_sum_buf


def _config_is_valid(config: LayerNorm2dConfig, params: LayerNorm2dParams):
    warp_c = divup(params.c, config.warps_c)
    block_x = config.warps_x * config.warp_x
    warps = config.warps_x * config.warps_c
    x = params.n * params.h * params.w
    smem_size = _smem_allocation(config, params)
    max_smem_size = 48 * 1024
    return (
        warps <= 8
        and warp_c >= min(64, params.c)
        and block_x <= x
        and smem_size < max_smem_size
    )


def _get_configs(
    params: LayerNorm2dParams, **_kwargs
) -> Generator[LayerNorm2dConfig, None, None]:
    warps_x_choices = [1, 2, 4, 8]
    warp_c_choices = [1, 2, 4, 8]
    warp_x_choices = [1, 2, 4, 8, 16, 32, 64]
    reverse_x_choices = [False, True]
    for warps_x, warps_c, warp_x, reverse_x in product(
        warps_x_choices, warp_c_choices, warp_x_choices, reverse_x_choices
    ):
        config = LayerNorm2dConfig(
            warps_x=warps_x, warps_c=warps_c, warp_x=warp_x, reverse_x=reverse_x
        )
        if _config_is_valid(config, params):
            yield config


KERNEL_NAME = "spio_layernorm_2d"

C_PER_REG = 64


def _get_specs(
    params: LayerNorm2dParams, config: LayerNorm2dConfig = None
) -> Tuple[List[gen.GenSpecs], LaunchParams]:
    """Get the specifications for the LayerNorm2d kernel."""
    params.validate()

    # Compute the number of warps and threads per block
    num_warps = config.warps_x * config.warps_c
    threads = num_warps * 32

    # Compute the number of blocks per grid
    x = params.n * params.h * params.w
    blocks = divup(x, config.block_x)

    # Generate the kernel launch parameters
    launch_params = LaunchParams(grid=blocks, block=threads)

    full_kernel_name = get_full_kernel_name(KERNEL_NAME, params)

    c8 = divup(params.c, 8)
    c2 = c8 * 4

    # Make warp_c a multiple of 64
    warp_c = divup(params.c, config.warps_c)
    warp_c = divup(warp_c, C_PER_REG) * C_PER_REG

    warp_c2 = divup(warp_c, 2)
    c2_per_thread = divup(warp_c, C_PER_REG)

    specs = [
        gen.Macro({"SPIO_LAYERNORM_2D_KERNEL": full_kernel_name}),
        gen.Fold("block_x", "x", config.block_x),
        gen.Fold("warp_c", "c", warp_c),
        gen.ParamsSpec(
            "Block",
            {
                "warps_x": config.warps_x,
                "warps_c": config.warps_c,
                "threads": threads,
                "blocks": blocks,
            },
        ),
        gen.ParamsSpec(
            "Params",
            {
                "eps": params.eps,
                "c2_per_thread": c2_per_thread,
                "warp_c2": warp_c2,
                "warp_x": config.warp_x,
                "has_weight": params.elementwise_affine,
                "has_bias": params.has_bias,
                "c": params.c,
            },
        ),
        gen.ParamsSpec(
            "Mode",
            {
                "reverse_x": config.reverse_x,
            },
        ),
        gen.Tensor("Input", "const uint4", {"x": x, "c8": c8}),
        gen.Tensor("Output", "__half2", {"x": x, "c2": c2}),
        gen.Tensor(
            "SmemInputStore", "uint4", {"ping_pong": 2, "x": config.warps_x, "c8": c8}
        ),
        gen.Tensor(
            "SmemInputLoad",
            "const __half2",
            {
                "ping_pong": 2,
                "x": config.warps_x,
                "c2": c2,
            },
        ),
        gen.Index(
            "ThreadIdx", {"warp_x": config.warps_x, "warp_c": config.warps_c, "c2": 32}
        ),
        gen.Tensor(
            "SmemSum", "float", {"warp_x": config.warps_x, "warp_c": config.warps_c}
        ),
    ]

    return specs, launch_params


layernorm_2d_kernel_factory = KernelFactory(
    LayerNorm2dParams,
    LayerNorm2dConfig,
    LayerNorm2dStats,
    kernel_name=KERNEL_NAME,
    configs=_get_configs,
    specs=_get_specs,
    kernel_source_file="layernorm_2d.cu",
    src_module="spio.src",
    includes_module="spio.include",
)
