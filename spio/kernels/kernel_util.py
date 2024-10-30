"""Utility functions for kernels."""

import torch


def get_first_device_in_args(args, default="cuda"):
    """Return the device of the first tensor argument.

    Return default if no arguments are torch.Tensor instances.
    """
    for arg in args:
        if isinstance(arg, torch.Tensor):
            return arg.device
    return default
