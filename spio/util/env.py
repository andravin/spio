"""Utility functions for environment variable handling."""

import os

TRUTHS = ["true", "yes", "y", "t"]


def env_var_is_true(env_var: str, default: str = "False") -> bool:
    """Returns True if the given environment variable is set to a truthy value.

    Positive integers are considered True, zero and negative integers are False.

    "Y", "y", "Yes", "yes", "T", "t", "True", "true", are also considered True.
    """
    value = os.environ.get(env_var, default).lower()
    if value in TRUTHS:
        return True
    if value.isdigit():
        return int(value) > 0
    return False
