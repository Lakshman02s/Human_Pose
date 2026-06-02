from __future__ import annotations

import cv2
import numpy as np

from models.base import PoseResult


COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_pose_frame(
    frame_bgr: np.ndarray,
    poses: list[PoseResult],
    fps: float,
    latency_ms: float,
    draw_bbox: bool = True,
    draw_keypoints: bool = True,
    draw_skeleton: bool = True,
    score_thr: float = 0.35,
    overlay_lines: list[str] | None = None,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    for pose in poses:
        bbox = pose.bbox_xyxy.astype(int)
        if draw_bbox:
            cv2.rectangle(canvas, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 4)
            label = f"ID {pose.track_id}" if pose.track_id is not None else "person"
            cv2.putText(canvas, label, (bbox[0], max(24, bbox[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        keypoints = pose.keypoints_xyc
        if draw_skeleton:
            for a, b in COCO_SKELETON:
                if keypoints[a, 2] >= score_thr and keypoints[b, 2] >= score_thr:
                    pt_a = tuple(keypoints[a, :2].astype(int))
                    pt_b = tuple(keypoints[b, :2].astype(int))
                    cv2.line(canvas, pt_a, pt_b, (255, 140, 0), 3)

        if draw_keypoints:
            for idx, kp in enumerate(keypoints):
                if kp[2] < score_thr:
                    continue
                cv2.circle(canvas, tuple(kp[:2].astype(int)), 5, (0, 0, 255), -1)
                cv2.putText(canvas, str(idx), tuple(kp[:2].astype(int) + 6), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

    text_lines = list(overlay_lines or [])
    text_lines.extend([
        f"FPS: {fps:.2f}",
        f"Latency: {latency_ms:.2f} ms",
        f"People: {len(poses)}",
    ])
    for idx, line in enumerate(text_lines):
        cv2.putText(canvas, line, (12, 32 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas
