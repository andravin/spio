from dataclasses import dataclass

import cupy as cp
import torch

from spio import (
    divup,
    GenDirectory,
    IndexSpec,
    TensorSpec,
    ParamsSpec,
    FragmentSpec,
    FragmentSpec,
    generate,
    compile_kernel,
    load_kernel,
)

from .auto_tune import auto_tune
from .kernel import Kernel
from .launch_params import LaunchParams
from .conv_small_group_params import ConvSmallGroupParams


@dataclass(frozen=True)
class ConvSmallGroupConfig:
    groups: int = 8
    block_p: int = 16

    def is_valid(self, params: ConvSmallGroupParams) -> bool:
        return True


class ConvSmallGroupKernel(Kernel):

    kernel_name = "conv_small_group"

    _FpropKernelsCache = {}
    _DgradKernelsCache = {}

    @classmethod
    def configs(cls, params: ConvSmallGroupParams):
        block_p_values = [
            block_p for block_p in [2, 4, 8, 16, 32, params.P] if block_p <= params.P
        ]
        groups_values = [
            groups
            for groups in [2, 4, 8, params.groups]
            if groups <= min(params.groups, 8)
        ]
        for groups in groups_values:
            for block_p in block_p_values:
                config = ConvSmallGroupConfig(groups=groups, block_p=block_p)
                if config.is_valid(params):
                    yield config

    @classmethod
    def fprop_kernel(
        cls,
        params: ConvSmallGroupParams,
        args,
        config: ConvSmallGroupConfig = None,
    ):
        if config is None:
            if params not in cls._FpropKernelsCache:
                kernel, _, _ = auto_tune(
                    cls,
                    params,
                    args,
                    igrad=False,
                )
                cls._FpropKernelsCache[params] = kernel
            return cls._FpropKernelsCache[params]
        else:
            return cls(params, config=config, igrad=False)

    @classmethod
    def dgrad_kernel(
        cls, params: ConvSmallGroupParams, args, config: ConvSmallGroupConfig = None
    ):
        if config is None:
            if params not in cls._DgradKernelsCache:
                kernel, _, _ = auto_tune(
                    cls,
                    params,
                    args,
                    igrad=True,
                )
                cls._DgradKernelsCache[params] = kernel
            return cls._DgradKernelsCache[params]
        else:
            return cls(params, config=config, igrad=True)

    @classmethod
    def arrange_kernel_args(cls, args, outputs=[]):
        return outputs + list(args[:2])

    def __init__(self, params, config=None, igrad=False, **kwargs):
        params.validate()
        super().__init__(**kwargs)

        R, S = params.R, params.S

        if igrad:
            N, C, H, W = params.N, params.C, params.P, params.Q
            P, Q = params.H, params.W
            PADDING_H, PADDING_W = (
                params.transpose_padding_h,
                params.transpose_padding_w,
            )
        else:
            N, C, H, W = params.N, params.C, params.H, params.W
            P, Q = params.P, params.Q
            PADDING_H, PADDING_W = params.padding_h, params.padding_w

        # Hardcoded parameter:
        GROUP_WIDTH = params.group_width

        # Derived parameters
        C8 = C // 8
        GROUPS = params.groups

        # Tiles
        BLOCK_P = min(config.block_p, P)
        BLOCK_Q = 16
        BLOCK_GROUPS = min(config.groups, GROUPS)

        # Derived Tiles
        BLOCK_C = BLOCK_GROUPS * GROUP_WIDTH
        BLOCK_C8 = BLOCK_C // 8
        BLOCK_W = BLOCK_Q + S - 1
        BLOCKS_N = N
        BLOCKS_P = divup(P, BLOCK_P)
        BLOCKS_Q = divup(Q, BLOCK_Q)
        BLOCKS_C8 = divup(C8, BLOCK_C8)
        BLOCKS = BLOCKS_N * BLOCKS_P * BLOCKS_Q * BLOCKS_C8
        WARPS = BLOCK_GROUPS
        THREADS = WARPS * 32

        self.launch_params = LaunchParams(grid=BLOCKS, block=THREADS)

        self.generate(
            [
                ParamsSpec(
                    "Block",
                    dict(
                        p=BLOCK_P,
                        q=BLOCK_Q,
                        c8=BLOCK_C8,
                        padding_h=PADDING_H,
                        padding_w=PADDING_W,
                        threads=THREADS,
                    ),
                ),
                ParamsSpec("Mode", dict(igrad=igrad)),
                IndexSpec(
                    "BlockIdx",
                    dict(n=BLOCKS_N, p=BLOCKS_P, q=BLOCKS_Q, c8=BLOCKS_C8),
                ),
                IndexSpec("InputIdx", dict(x=BLOCK_W, c8=BLOCK_C8)),
                TensorSpec("Input", "const uint4", dict(n=N, y=H, x=W, c8=C8)),
                TensorSpec("Output", "uint4", dict(n=N, p=P, q=Q, k8=C8)),
                TensorSpec("Weights", "const uint4", dict(k=C, r=R, s=S)),
                TensorSpec("SmemWeights", "uint4", dict(k=BLOCK_C, r=R, s=S)),
                TensorSpec(
                    "ConstSmemWeights",
                    "const uint4",
                    dict(kd8=BLOCK_C8, km8=8, rs=R * S),
                ),
                IndexSpec("SmemWeightsLoadIdx", dict(kd8=BLOCK_C8, rs=4, km8=8)),
                TensorSpec(
                    "SmemInput",
                    "uint4",
                    dict(ping_pong=2, x=BLOCK_W, c8=BLOCK_C8 + 1),
                ),
                IndexSpec(
                    "SmemInputLoadIdx",
                    dict(c8=BLOCK_C8, repeat=32 // BLOCK_Q, q=BLOCK_Q),
                ),
                IndexSpec("SmemOutputStoreIdx", dict(k8=BLOCK_C8, lane=32)),
                TensorSpec(
                    "SmemOutput",
                    "__half2",
                    dict(q=BLOCK_Q, k8=BLOCK_C8 + 1, k2=4),
                ),
                TensorSpec(
                    "ConstSmemOutput",
                    "const uint4",
                    dict(q=BLOCK_Q, k8=BLOCK_C8 + 1),
                ),
                IndexSpec("OutputStoreIdx", dict(q=BLOCK_Q, k8=BLOCK_C8)),
                FragmentSpec("Acc", "MMA_M16_N8_F32_C", "q", "k"),
            ]
        )
