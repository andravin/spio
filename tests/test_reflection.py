import torch

from spio.reflection import get_kernel_reflection
from spio.kernels import MlpParams


def test_kernel_reflection_mlp_small_c():
    """Test the reflection for the mlp_tiny_c kernel."""
    ref = get_kernel_reflection("mlp_tiny_c")
    assert ref.kernel_name == "mlp_tiny_c"

    # Make arguments and check their types and shapes.
    params = MlpParams(x=128, c=64, r=256, k=64)
    args = ref.make_args(params=params)
    exp_weight = args["exp_weight"]
    assert exp_weight.dtype == torch.float16
    assert exp_weight.shape == (params.c // 16, params.r, 16)
    prj_weight = args["prj_weight"]
    assert prj_weight.dtype == torch.float16
    assert prj_weight.shape == (params.r // 16, params.k, 16)
    exp_bias = args["exp_bias"]
    assert exp_bias.dtype == torch.float32
    assert exp_bias.shape == (params.r,)
    prj_bias = args["prj_bias"]
    assert prj_bias.dtype == torch.float32
    assert prj_bias.shape == (params.k,)
    inputs = args["input"]
    assert inputs.dtype == torch.float16
    assert inputs.shape == (params.x, params.c)
    outputs = args["output"]
    assert outputs.dtype == torch.float16
    assert outputs.shape == (params.x, params.k)
