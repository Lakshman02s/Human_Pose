from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VisualizationConfig:
    draw_bbox: bool = True
    draw_keypoints: bool = True
    draw_skeleton: bool = True
    draw_scores: bool = True
    bbox_color: tuple[int, int, int] = (0, 255, 0)
    text_color: tuple[int, int, int] = (255, 255, 255)


@dataclass
class RuntimeConfig:
    det_score_thr: float = 0.5
    kpt_score_thr: float = 0.35
    device: str = "auto"
    use_fp16: bool = False
    batch_size: int = 1
    enable_tracking: bool = True
    enable_smoothing: bool = True
    tracking_iou_thr: float = 0.3
    smoothing_alpha: float = 0.6


@dataclass
class InputConfig:
    source: str
    source_type: str = "video"
    output_path: Optional[Path] = None
    display: bool = True
    save_json: bool = True


@dataclass
class ModelPaths:
    det_config: str
    det_checkpoint: str
    pose_config: str
    pose_checkpoint: str


@dataclass
class BenchmarkConfig:
    warmup_frames: int = 30
    max_frames: int = 300
    save_screenshots: bool = True
