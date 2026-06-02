from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass

import psutil
import torch


class FPSMeter:
    def __init__(self, window_size: int = 30):
        self.timestamps = deque(maxlen=window_size)

    def update(self) -> float:
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) < 2:
            return 0.0
        delta = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / delta if delta > 0 else 0.0


class RollingStats:
    def __init__(self, maxlen: int = 512):
        self.values = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        self.values.append(float(value))

    def summary(self) -> dict:
        if not self.values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        ordered = sorted(self.values)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
        return {
            "count": len(self.values),
            "mean": statistics.fmean(self.values),
            "p50": p50,
            "p95": p95,
            "max": max(self.values),
        }


@dataclass
class SystemSnapshot:
    cpu_percent: float
    ram_percent: float
    gpu_memory_mb: float


def sample_system_usage(device: str) -> SystemSnapshot:
    cpu_percent = psutil.cpu_percent(interval=None)
    ram_percent = psutil.virtual_memory().percent
    gpu_memory_mb = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return SystemSnapshot(cpu_percent=cpu_percent, ram_percent=ram_percent, gpu_memory_mb=gpu_memory_mb)
