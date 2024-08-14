import torch
from spio.kernels import ConvSmallGroupKernel, ConvSmallGroupWgradKernel


def run_kernel_test(kernel_class, params):
    kernel = kernel_class(params)
    outputs, *args = kernel.get_test_args()
    kernel(outputs, *args)
    torch_outputs = kernel.reference(*args)
    torch.testing.assert_close(outputs, torch_outputs, msg=str(params))


def run_kernel_tests(kernel_class, test_params):
    for params in test_params:
        run_kernel_test(kernel_class, params)


def run_igrad_kernel_test(kernel_class, params):
    kernel = ConvSmallGroupKernel(params, igrad=True)
    inputs, weights, deltas = kernel.get_grad_test_args()
    igrad_ref, _ = kernel.grad_reference(inputs, weights, deltas)
    igrad = torch.zeros_like(inputs)
    kernel(igrad, deltas.detach(), weights.detach())
    torch.testing.assert_close(igrad, igrad_ref, msg=str(params))


def run_wgrad_kernel_test(kernel_class, params):
    kernel = ConvSmallGroupWgradKernel(params)
    inputs, weights, deltas = kernel.get_grad_test_args()
    _, wgrad_ref = kernel.grad_reference(inputs, weights, deltas)
    wgrad = torch.zeros_like(weights)
    TOLERANCE = 1e-3
    depth = params.N * params.P * params.Q
    tolerance = TOLERANCE * depth
    max_val = torch.amax(torch.abs(wgrad_ref))
    kernel(wgrad, inputs.detach(), deltas.detach())
    torch.testing.assert_close(
        wgrad, wgrad_ref, msg=str(params), rtol=TOLERANCE, atol=tolerance
    )


def run_igrad_kernel_tests(kernel_class, test_params):
    for params in test_params:
        run_igrad_kernel_test(kernel_class, params)


def run_wgrad_kernel_tests(kernel_class, test_params):
    for params in test_params:
        run_wgrad_kernel_test(kernel_class, params)


def test_benchmark_conv_small_group():
    run_kernel_test(
        ConvSmallGroupKernel, ConvSmallGroupKernel.Params(N=128, C=128, H=64, W=64)
    )


def test_benchmark_conv_small_group_5x5():
    run_kernel_test(
        ConvSmallGroupKernel,
        ConvSmallGroupKernel.Params(N=128, C=128, H=64, W=64, R=5, S=5, padding=2),
    )


def test_benchmark_igrad_conv_small_group():
    run_igrad_kernel_test(
        ConvSmallGroupKernel, ConvSmallGroupKernel.Params(N=128, C=128, H=64, W=64)
    )


def test_benchmark_wgrad_conv_small_group():
    run_wgrad_kernel_test(
        ConvSmallGroupWgradKernel,
        ConvSmallGroupWgradKernel.Params(N=128, C=1024, H=64, W=64),
    )


def test_benchmark_wgrad_conv_small_group_16h_16w():
    run_wgrad_kernel_test(
        ConvSmallGroupWgradKernel,
        ConvSmallGroupWgradKernel.Params(N=128, C=1024, H=16, W=16),
    )


def test_wgrad_conv_small_group():
    run_wgrad_kernel_tests(ConvSmallGroupWgradKernel, _get_conv_small_group_params())


def test_kernel_conv_small_group():
    run_kernel_tests(ConvSmallGroupKernel, _get_conv_small_group_params())


def test_conv_small_group_igrad():
    run_igrad_kernel_tests(ConvSmallGroupKernel, _get_conv_small_group_params())


def _get_conv_small_group_params():
    Params = ConvSmallGroupKernel.Params
    conv_rxs_tests = [
        Params(N=1, C=128, H=32, W=32, R=r, S=s, padding=(r // 2, s // 2))
        for r in range(1, 6)
        for s in range(1, 6)
    ]

    padding_tests = [
        Params(N=1, C=128, H=64, W=64, padding=0),
        Params(N=4, C=128, H=20, W=19, padding=0),
        Params(N=1, C=128, H=64, W=64, padding=2),
        Params(N=1, C=128, H=64, W=64, padding=7),
    ]

    MIN_GROUPS = 1
    MAX_GROUPS = 16
    group_tests = [
        Params(N=4, C=groups * 8, H=32, W=32)
        for groups in range(MIN_GROUPS, MAX_GROUPS + 1)
    ]

    misc_tests = [
        Params(N=6, C=128, H=49, W=33),
        Params(N=6, C=128, H=6, W=5),
        Params(N=128, C=1024, H=7, W=7),
        Params(N=128, C=64, H=64, W=64),
        Params(N=128, C=128, H=64, W=64),
        Params(N=128, C=256, H=64, W=64),
        Params(N=3, C=128, H=16, W=16),
        Params(N=4, C=128, H=32, W=16),
        Params(N=5, C=128, H=16, W=32),
        Params(N=6, C=128, H=48, W=32),
    ]

    all_tests = conv_rxs_tests + padding_tests + group_tests + misc_tests
    return all_tests
