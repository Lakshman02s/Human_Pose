from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    device: str
    cuda_available: bool
    gpu_name: str | None
    cuda_version: str | None
    fp16_supported: bool


def select_device(preferred: str = "auto") -> DeviceInfo:
    if preferred != "auto":
        device = preferred
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    cuda_available = device.startswith("cuda") and torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    cuda_version = torch.version.cuda if cuda_available else None
    fp16_supported = cuda_available
    return DeviceInfo(
        device=device,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        fp16_supported=fp16_supported,
    )
