import torch


def get_formatted_device_name(device: str) -> str:
    with torch.device(device) as device_obj:
        device_name = torch.cuda.get_device_name(device_obj)
        return device_name.replace(" ", "_").replace("(", "_").lower()

def get_formatted_arch(device: str) -> str:
    with torch.device(device) as device_obj:
        arch = "sm_{0}{1}".format(*torch.cuda.get_device_capability(device=device_obj))
        return arch
