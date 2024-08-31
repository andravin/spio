import torch


def assert_all_close(actual, expected, msg=None, rtol=2.0**-9):
    actual = actual.float()
    max_abs_expected = torch.amax(expected)
    tol = rtol * max_abs_expected
    diff = actual - expected
    absdiff = torch.abs(diff)
    if not torch.all(absdiff <= tol):
        baddies = torch.nonzero(absdiff > tol, as_tuple=True)
        bad_absdiff = absdiff[baddies]
        bad_expected = expected[baddies]
        bad_actual = actual[baddies]
        m = f"Tensors not close: tolerance={tol}\n"
        max_errors = 30
        for a, e, ad in zip(
            bad_actual[:max_errors], bad_expected[:max_errors], bad_absdiff[:max_errors]
        ):
            ratio = ad / abs(e)
            m += f"actual ={a:>10.6f} expected ={e:>10.6f} |diff| = {ad:>10.6f} |diff / expected| = {ratio:>10.6f}\n"
        if len(bad_actual) > max_errors:
            m += f"... and {len(bad_actual) - max_errors} more\n"
        if msg is not None:
            m += " " + msg
        raise AssertionError(m)
