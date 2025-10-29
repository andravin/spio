"""Functions for getting CUDA device information."""

import torch


def get_formatted_device_name(device: str) -> str:
    """Get a formatted name of the CUDA device."""
    with torch.device(device) as device_obj:
        device_name = torch.cuda.get_device_name(device_obj)
        return device_name.replace(" ", "_").replace("(", "_").lower()


def get_formatted_arch(device: str) -> str:
    """Get the architecture of the CUDA device."""
    with torch.device(device) as device_obj:
        cap = torch.cuda.get_device_capability(device=device_obj)
        arch = f"sm_{cap[0]}{cap[1]}"
        return arch


def get_device_ordinal(device: str) -> int:
    """Get the device ordinal for a given PyTorch device str.

    Args:
        device (str): The device string (e.g., "cuda:0").

    Returns:
        int: The device ordinal.
    """
    device_obj = torch.device(device)
    idx = device_obj.index
    if idx is None:
        idx = 0
    return idx
