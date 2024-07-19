import torch
from spio.kernels import ConvSmallGroupKernel


def run_kernel_test(kernel_class, params):
    kernel = kernel_class(params)
    outputs, *args = kernel.get_test_args()
    kernel(outputs, *args)
    torch_outputs = kernel.reference(*args)
    torch.testing.assert_close(outputs, torch_outputs, msg=str(params))


def run_kernel_tests(kernel_class, test_params):
    for params in test_params:
        run_kernel_test(kernel_class, params)


def test_benchmark_conv_small_group():
    run_kernel_test(
        ConvSmallGroupKernel, ConvSmallGroupKernel.Params(N=128, C=128, H=64, W=64)
    )


def test_benchmark_conv_small_group_5x5():
    run_kernel_test(
        ConvSmallGroupKernel,
        ConvSmallGroupKernel.Params(N=128, C=128, H=64, W=64, R=5, S=5, padding=1),
    )


def test_kernel_conv_small_group():

    conv_rxs_tests = [
        ConvSmallGroupKernel.Params(
            N=1, C=128, H=32, W=32, R=r, S=s, padding=(r // 2, s // 2)
        )
        for r in range(1, 6)
        for s in range(1, 6)
    ]

    padding_tests = [
        ConvSmallGroupKernel.Params(N=1, C=128, H=64, W=64, padding=0),
        ConvSmallGroupKernel.Params(N=4, C=128, H=20, W=19, padding=0),
        ConvSmallGroupKernel.Params(N=1, C=128, H=64, W=64, padding=2),
        ConvSmallGroupKernel.Params(N=1, C=128, H=64, W=64, padding=7),
    ]

    MIN_GROUPS = 1
    MAX_GROUPS = 16
    group_tests = [
        ConvSmallGroupKernel.Params(N=4, C=groups * 8, H=32, W=32)
        for groups in range(MIN_GROUPS, MAX_GROUPS + 1)
    ]

    misc_tests = [
        ConvSmallGroupKernel.Params(N=6, C=128, H=49, W=33),
        ConvSmallGroupKernel.Params(N=6, C=128, H=6, W=5),
        ConvSmallGroupKernel.Params(N=128, C=1024, H=7, W=7),
        ConvSmallGroupKernel.Params(N=128, C=64, H=64, W=64),
        ConvSmallGroupKernel.Params(N=128, C=128, H=64, W=64),
        ConvSmallGroupKernel.Params(N=128, C=256, H=64, W=64),
        ConvSmallGroupKernel.Params(N=3, C=128, H=16, W=16),
        ConvSmallGroupKernel.Params(N=4, C=128, H=32, W=16),
        ConvSmallGroupKernel.Params(N=5, C=128, H=16, W=32),
        ConvSmallGroupKernel.Params(N=6, C=128, H=48, W=32),
    ]

    all_tests = conv_rxs_tests + padding_tests + group_tests + misc_tests

    run_kernel_tests(ConvSmallGroupKernel, all_tests)
