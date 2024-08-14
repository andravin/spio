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
    padding: int = 1  # Also allows tuple (padding_h, padding_w)
    R: int = 3
    S: int = 3

    @property
    def GROUP_WIDTH(self):
        return 8
    
    @property
    def GROUPS(self):
        return self.C // self.GROUP_WIDTH

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
        assert self.C % self.GROUP_WIDTH == 0
        assert self.R in range(6)
        assert self.S in range(6)

    def get_test_args(self):
        N, C, H, W = self.N, self.C, self.H, self.W
        P, Q = self.P, self.Q
        R, S = self.R, self.S

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

    def get_grad_test_args(self):
        outputs, inputs, weights = self.get_test_args()
        inputs.requires_grad = True
        weights.requires_grad = True
        deltas = torch.randn_like(outputs)
        return (inputs, weights, deltas)

    def reference(self, inputs, weights):
        return torch.nn.functional.conv2d(
            inputs,
            weights,
            bias=None,
            stride=1,
            padding=self.padding,
            groups=self.GROUPS,
        )

    def grad_reference(self, inputs, weights, deltas):
        """https://discuss.pytorch.org/t/compare-correct-gradients-of-pytorch-with-own-implementation/177152/2"""
        outputs = self.reference(inputs, weights)
        wgrad = torch.autograd.grad(outputs, weights, deltas, retain_graph=True)
        igrad = torch.autograd.grad(outputs, inputs, deltas)
        return (igrad[0], wgrad[0])


class ConvSmallGroupKernel:

    Params = _ConvSmallGroupParams

    def __init__(self, params: Params, debug=False, lineinfo=True, igrad=False):
        self.params = params

        R, S = params.R, params.S

        if igrad:
            N, C, H, W = params.N, params.C, params.P, params.Q
            P, Q = params.H, params.W
            PADDING_H = R - 1 - params.padding_h
            PADDING_W = S - 1 - params.padding_w
        else:
            N, C, H, W = params.N, params.C, params.H, params.W
            P, Q = params.P, params.Q
            PADDING_H, PADDING_W = params.padding_h, params.padding_w


        # Hardcoded parameter:
        GROUP_WIDTH = params.GROUP_WIDTH

        # Derived parameters
        C8 = C // 8
        GROUPS = params.GROUPS

        # Tile parameters
        # These could be optimized by auto-tuning.
        target_groups = 8
        target_block_p = 16

        # Tiles
        BLOCK_P = min(target_block_p, P)
        BLOCK_Q = 16
        BLOCK_GROUPS = min(target_groups, GROUPS)

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
        return self.params.get_test_args()

    def get_grad_test_args(self):
        return self.params.get_grad_test_args()

    def reference(self, inputs, weights):
        return self.params.reference(inputs, weights)

    def grad_reference(self, inputs, weights, deltas):
        return self.params.grad_reference(inputs, weights, deltas)
