from dataclasses import dataclass
from tempfile import NamedTemporaryFile

import cupy as cp
import torch

from .conv_small_group import ConvSmallGroupKernel
from .auto_tune import auto_tune
from .kernel import Kernel
from .launch_params import LaunchParams
from .conv_small_group_params import ConvSmallGroupParams

from spio import (
    divup,
    GenDirectory,
    generate,
    ParamsSpec,
    IndexSpec,
    TensorSpec,
    FragmentSpec,
)


@dataclass(frozen=True)
class ConvSmallGroupWgradConfig:
    groups: int = 8
    block_h: int = 16

    def is_valid(self, params: ConvSmallGroupParams) -> bool:
        return True


class ConvSmallGroupWgradKernel(Kernel):

    kernel_name = "conv_small_group_wgrad"

    _KernelsCache = {}

    @classmethod
    def configs(cls, params: ConvSmallGroupParams):
        block_h_values = [
            block_h for block_h in [2, 4, 8, 16, 32, params.H] if block_h <= params.H
        ]
        groups_values = [
            groups
            for groups in [2, 4, 8, params.groups]
            if groups <= min(params.groups, 8)
        ]
        for groups in groups_values:
            for block_h in block_h_values:
                config = ConvSmallGroupWgradConfig(groups=groups, block_h=block_h)
                if config.is_valid(params):
                    yield config

    @classmethod
    def wgrad_kernel(cls, params: ConvSmallGroupParams, args, config=None):
        if config is None:
            if params not in cls._KernelsCache:
                kernel, _, _ = auto_tune(cls, params, args)
                cls._KernelsCache[params] = kernel
            return cls._KernelsCache[params]
        else:
            return cls(params, config=config)

    def __init__(self, params, config=None, **kwargs):
        params.validate()
        super().__init__(**kwargs)

        R, S = params.R, params.S

        N, C, H, W = params.N, params.C, params.H, params.W
        P, Q = params.P, params.Q
        PADDING_H, PADDING_W = params.padding_h, params.padding_w
        TRANSPOSE_PADDING_H = params.transpose_padding_h

        # Hardcoded parameters:
        GROUP_WIDTH = params.group_width

        # Derived parameters
        C8 = C // 8
        GROUPS = params.groups

        # Tiles
        BLOCK_H = min(config.block_h, H)
        BLOCK_Q = 8
        BLOCK_GROUPS = min(config.groups, GROUPS)

        # Derived Tiles
        BLOCK_C = BLOCK_GROUPS * GROUP_WIDTH
        BLOCK_C8 = BLOCK_C // 8
        BLOCK_W = BLOCK_Q + S - 1
        BLOCK_P = BLOCK_H + R - 1
        BLOCKS_N = N
        BLOCKS_H = divup(H, BLOCK_H)
        BLOCKS_Q = divup(Q, BLOCK_Q)
        BLOCKS_C8 = divup(C8, BLOCK_C8)
        BLOCKS = BLOCKS_N * BLOCKS_H * BLOCKS_Q * BLOCKS_C8
        WARPS = BLOCK_GROUPS
        THREADS = WARPS * 32
        S2_UP = divup(S, 2)

        SMEM_TENSORS = [
            TensorSpec(
                "SmemInput",
                "uint4",
                dict(ping_pong=2, x=BLOCK_W, c8=BLOCK_C8 + 1),
            ),
            TensorSpec(
                "SmemDelta",
                "uint4",
                dict(ping_pong=2, q=BLOCK_Q, k8=BLOCK_C8 + 1),
            ),
            TensorSpec("SmemWgrad", "float2", dict(k8=BLOCK_C8, s=S, c=8, k2=4)),
        ]

        # TODO: ensure that the smem tensors fit in the shared memory.
        smem_size = sum(tensor.num_bytes for tensor in SMEM_TENSORS)

        self.launch_params = LaunchParams(grid=BLOCKS, block=THREADS)

        self.generate(
            [
                #
                # Block parameters.
                #
                ParamsSpec(
                    "Block",
                    dict(
                        h=BLOCK_H,
                        q=BLOCK_Q,
                        c8=BLOCK_C8,
                        p=BLOCK_P,
                        threads=THREADS,
                    ),
                ),
                ParamsSpec(
                    "Params",
                    dict(
                        R=R,
                        S=S,
                        PADDING_H=PADDING_H,
                        PADDING_W=PADDING_W,
                        TRANSPOSE_PADDING_H=TRANSPOSE_PADDING_H,
                    ),
                ),
                IndexSpec(
                    "BlockIdx",
                    dict(n=BLOCKS_N, y=BLOCKS_H, q=BLOCKS_Q, c8=BLOCKS_C8),
                ),
                #
                # Input loading.
                #
                IndexSpec("InputIdx", dict(x=BLOCK_W, c8=BLOCK_C8)),
                TensorSpec("Input", "const uint4", dict(n=N, y=H, x=W, c8=C8)),
                IndexSpec(
                    "SmemInputLoadIdx",
                    dict(c8=BLOCK_C8, repeat=32 // (2 * BLOCK_Q), s=2, q=BLOCK_Q),
                ),
                #
                # Delta loading
                #
                IndexSpec("DeltaIdx", dict(q=BLOCK_Q, k8=BLOCK_C8)),
                TensorSpec("Delta", "const uint4", dict(n=N, p=P, q=Q, k8=C8)),
                IndexSpec(
                    "SmemDeltaLoadIdx",
                    dict(k8=BLOCK_C8, repeat=32 // BLOCK_Q, q=BLOCK_Q),
                ),
                #
                # Accumulator
                #
                FragmentSpec("Acc", "MMA_M16_N8_F32_C", "c", "k"),
                TensorSpec("AccTensor", "spio::MMA_M16_N8_F32_C", dict(s2=S2_UP, r=R)),
                #
                # Weights storing.
                #
                IndexSpec("SmemWgradStoreIdx", dict(k8=BLOCK_C8, lane=32)),
                # Each thread stores 8k for a particular (k8, r, s, c).
                IndexSpec("WgradStoreIdx", dict(k8=BLOCK_C8, s=S, c=8)),
                # Reduce Wgrad through global memory using float32 precision.
                TensorSpec("Wgrad", "float", dict(k=C, r=R, s=S, c=8)),
            ]
            + SMEM_TENSORS,
        )

    def __call__(self, wgrad, inputs, deltas):
        # Zero out the wgrad tensor.
        assert inputs.dtype == torch.float16
        assert deltas.dtype == torch.float16
        wgrad_f32 = torch.zeros_like(wgrad, dtype=torch.float32)
        self.launch(wgrad_f32, inputs, deltas)
        wgrad.copy_(wgrad_f32)
