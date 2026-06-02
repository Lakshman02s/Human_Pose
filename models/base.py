from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    bbox_xyxy: np.ndarray
    score: float
    label: int = 0


@dataclass
class PoseResult:
    bbox_xyxy: np.ndarray
    score: float
    keypoints_xyc: np.ndarray
    track_id: int | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    frame_id: int
    poses: list[PoseResult]
    detector_ms: float
    pose_ms: float
    total_ms: float
    rendered_frame: np.ndarray | None = None
