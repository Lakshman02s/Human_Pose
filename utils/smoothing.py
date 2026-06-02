from __future__ import annotations

import numpy as np


class PoseSmoother:
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.state: dict[int, np.ndarray] = {}

    def apply(self, track_id: int, keypoints_xyc: np.ndarray) -> np.ndarray:
        if track_id not in self.state:
            self.state[track_id] = keypoints_xyc.copy()
            return keypoints_xyc
        smoothed = self.alpha * keypoints_xyc + (1.0 - self.alpha) * self.state[track_id]
        self.state[track_id] = smoothed
        return smoothed
