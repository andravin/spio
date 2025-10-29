"""Functions for asserting that two tensors are close to each other."""

import torch


def assert_all_close_with_acc_depth(
    actual: torch.Tensor,
    expected: torch.Tensor,
    msg: str = None,
    acc_depth: int = None,
    abs_mean: float = 1.6,
    atol_fudge: float = 3.5,
    rtol_fudge: float = 2.0,
):
    """Check tensor proximity using accumulation depth.

    Computes the absolute tolerance as a function of the accumumlation depth.

    Note: this function assumes that operands are in float16 and accumulation
    is in float32.

    Args:
        actual: The actual tensor.
        expected: The expected tensor.
        msg: Optional message for assertion failure.
        acc_depth: Accumulation depth for scaling tolerances.
        abs_mean: Expected mean absolute value of the tensor elements.
        atol_fudge: Fudge factor for the absolute tolerance.
        rtol_fudge: Fudge factor for the relative tolerance.
    """
    float16_precision = 5e-4
    float32_precision = 1.19e-7
    atol = acc_depth * abs_mean * abs_mean * float32_precision * atol_fudge
    rtol = float16_precision * rtol_fudge
    assert_all_close(actual, expected, atol=atol, rtol=rtol, msg=msg)


def assert_all_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 0,
    rtol: float = 0,
    msg: str = None,
    max_errors: int = 128,
):
    """Assert that two tensors are close to each other."""
    expected = expected.float()
    actual = actual.float()
    absdiff = torch.abs(actual - expected)
    absdiff_tol = atol + rtol * expected.abs()
    if not torch.all(absdiff <= absdiff_tol):
        baddies = torch.nonzero(absdiff > absdiff_tol, as_tuple=True)
        bad_absdiff = absdiff[baddies]
        bad_expected = expected[baddies]
        bad_actual = actual[baddies]
        bad_absdiff_tol = absdiff_tol[baddies]
        m = f"Tensors not close: atol={atol}, rtol={rtol}\n"
        bad_indices = [tuple(int(x) for x in v) for v in zip(*baddies)]
        for a, e, ad, adt, idx in zip(
            bad_actual[:max_errors],
            bad_expected[:max_errors],
            bad_absdiff[:max_errors],
            bad_absdiff_tol[:max_errors],
            bad_indices[:max_errors],
        ):
            m += (
                f"{list(idx)} actual ={a:>10.6f} expected ={e:>10.6f} |diff| = {ad:>10.6f} > "
                f"abs_diff_tol = {adt:>10.6f}\n"
            )
        if len(bad_actual) > max_errors:
            m += f"... and {len(bad_actual) - max_errors} more\n"
        if msg is not None:
            m += " " + msg
        raise AssertionError(m)
