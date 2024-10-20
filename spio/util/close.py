import torch


def assert_all_close_with_acc_depth(actual, expected, msg=None, acc_depth=None, abs_mean=1.6, fudge=2.5):
    float16_precision = 5e-4
    float32_precision = 1.19e-7 
    atol = acc_depth * abs_mean * abs_mean * float32_precision
    # We see rounding errors larger than float16 precision in some cases.
    # Let's fudge it a bit.
    rtol = float16_precision * fudge
    assert_all_close(actual, expected, atol=atol, rtol=rtol, msg=msg)


def assert_all_close(actual, expected, atol=0, rtol=0, msg=None):
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
        max_errors = 30
        for a, e, ad, adt in zip(
            bad_actual[:max_errors], bad_expected[:max_errors], bad_absdiff[:max_errors], bad_absdiff_tol[:max_errors]
        ):
            ratio = ad / abs(e)
            m += f"actual ={a:>10.6f} expected ={e:>10.6f} |diff| = {ad:>10.6f} > abs_diff_tol = {adt:>10.6f}\n"
        if len(bad_actual) > max_errors:
            m += f"... and {len(bad_actual) - max_errors} more\n"
        if msg is not None:
            m += " " + msg
        raise AssertionError(m)
