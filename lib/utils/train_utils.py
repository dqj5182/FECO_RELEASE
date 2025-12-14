import torch


def move_to_device(d, device):
    """
    Recursively moves all torch.Tensor inside dict (or nested dicts) to device.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                move_to_device(v, device)
            elif isinstance(v, torch.Tensor):
                d[k] = v.to(device)
    elif isinstance(d, torch.Tensor):
        return d.to(device)