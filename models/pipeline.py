from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from models.base import PipelineResult, PoseResult
from models.human_detector import HumanDetector
from models.pose_estimator import PoseEstimator
from utils.smoothing import PoseSmoother
from utils.tracking import IoUTracker
from utils.visualization import draw_pose_frame


@dataclass
class PipelineOptions:
    keypoint_score_thr: float = 0.35
    draw_bbox: bool = True
    draw_keypoints: bool = True
    draw_skeleton: bool = True
    enable_tracking: bool = True
    enable_smoothing: bool = True
    tracking_iou_thr: float = 0.3
    smoothing_alpha: float = 0.6


class PosePipeline:
    def __init__(self, detector: HumanDetector, estimator: PoseEstimator, options: PipelineOptions):
        self.detector = detector
        self.estimator = estimator
        self.options = options
        self.tracker = IoUTracker(iou_threshold=options.tracking_iou_thr) if options.enable_tracking else None
        self.smoother = PoseSmoother(alpha=options.smoothing_alpha) if options.enable_smoothing else None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        fps: float = 0.0,
        overlay_lines: list[str] | None = None,
    ) -> PipelineResult:
        detections, det_ms = self.detector.detect(frame_bgr)
        bboxes = [det.bbox_xyxy for det in detections]
        poses, pose_ms = self.estimator.predict(frame_bgr, bboxes)

        if self.tracker is not None and poses:
            track_ids = self.tracker.update([pose.bbox_xyxy for pose in poses])
            for pose, track_id in zip(poses, track_ids):
                pose.track_id = track_id
                if self.smoother is not None:
                    pose.keypoints_xyc = self.smoother.apply(track_id, pose.keypoints_xyc)

        total_ms = det_ms + pose_ms
        rendered = draw_pose_frame(
            frame_bgr,
            poses,
            fps=fps,
            latency_ms=total_ms,
            draw_bbox=self.options.draw_bbox,
            draw_keypoints=self.options.draw_keypoints,
            draw_skeleton=self.options.draw_skeleton,
            score_thr=self.options.keypoint_score_thr,
            overlay_lines=overlay_lines,
        )
        return PipelineResult(
            frame_id=frame_id,
            poses=poses,
            detector_ms=det_ms,
            pose_ms=pose_ms,
            total_ms=total_ms,
            rendered_frame=rendered,
        )

    def process_batch(self, frames_bgr: Iterable[np.ndarray], start_frame_id: int = 0) -> list[PipelineResult]:
        results: list[PipelineResult] = []
        for offset, frame_bgr in enumerate(frames_bgr):
            results.append(self.process_frame(frame_bgr, frame_id=start_frame_id + offset))
        return results


def poses_to_json_serializable(poses: list[PoseResult]) -> list[dict]:
    serializable = []
    for pose in poses:
        serializable.append(
            {
                "track_id": pose.track_id,
                "score": pose.score,
                "bbox_xyxy": pose.bbox_xyxy.tolist(),
                "keypoints_xyc": pose.keypoints_xyc.tolist(),
            }
        )
    return serializable
