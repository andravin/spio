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
)


@dataclass(frozen=True)
class _ConvSmallGroupParams:
    N: int
    C: int
    H: int
    W: int
    padding: int = 1 # Also allow tuple (padding_h, padding_w)
    R: int = 3
    S: int = 3

    @property
    def padding_h(self):
        return self.padding[0] if isinstance(self.padding, tuple) else self.padding

    @property
    def padding_w(self):
        return self.padding[1] if isinstance(self.padding, tuple) else self.padding

    @property
    def P(self):
        return self.H + 2 * self.padding_h - self.R + 1

    @property
    def Q(self):
        return self.W + 2 * self.padding_w - self.S + 1

    def __post_init__(self):
        assert self.N > 0
        assert self.C > 0
        assert self.H > 0
        assert self.W > 0
        assert self.padding_h >= 0
        assert self.padding_w >= 0
        assert self.C % 8 == 0
        assert self.R in range(6)
        assert self.S in range(6)


class ConvSmallGroupKernel:

    Params = _ConvSmallGroupParams

    def __init__(self, params: Params, debug=False, lineinfo=True):
        self.params = params

        N, C, H, W = params.N, params.C, params.H, params.W

        PADDING_H, PADDING_W = params.padding_h, params.padding_w

        R, S = params.R, params.S

        P = params.P
        Q = params.Q

        # Hardcoded parameter:
        GROUP_WIDTH = 8

        # Derived parameters
        C8 = C // 8
        GROUPS = C // GROUP_WIDTH

        # Tiles
        BLOCK_P = min(16, P)
        BLOCK_Q = 16
        BLOCK_GROUPS = min(8, GROUPS)

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

        self._blocks = BLOCKS
        self._threads = THREADS
        self.GROUPS = GROUPS
        self.GROUP_WIDTH = GROUP_WIDTH

        with GenDirectory() as include_dir:
            generate(
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
                ],
                include_dir / "parameters.h",
            )

            self.module, self.kernel = compile_kernel(
                kernel_name="conv_small_group",
                debug=debug,
                lineinfo=lineinfo,
                includes=[include_dir],
            )

    def __call__(self, outputs, inputs, weights):
        cp_inputs = cp.asarray(inputs)
        cp_outputs = cp.asarray(outputs)
        cp_weights = cp.asarray(weights)
        self.kernel(
            (self._blocks,), (self._threads,), (cp_outputs, cp_inputs, cp_weights)
        )

    def get_test_args(self):
        N, C, H, W = self.params.N, self.params.C, self.params.H, self.params.W
        P, Q = self.params.P, self.params.Q
        R, S = self.params.R, self.params.S

        inputs = torch.randn((N, C, H, W), device="cuda", dtype=torch.float16).to(
            memory_format=torch.channels_last
        )

        weights = torch.randn(
            (C, self.GROUP_WIDTH, R, S), device="cuda", dtype=torch.float16
        ).to(memory_format=torch.channels_last)

        outputs = torch.zeros((N, C, P, Q), device="cuda", dtype=torch.float16).to(
            memory_format=torch.channels_last
        )
        return (outputs, inputs, weights)

    def reference(self, inputs, weights):
        return torch.nn.functional.conv2d(
            inputs,
            weights,
            bias=None,
            stride=1,
            padding=self.params.padding,
            groups=self.GROUPS,
        )
