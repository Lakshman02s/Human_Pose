from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    track_id: int
    bbox_xyxy: np.ndarray
    age: int = 0


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, bboxes_xyxy: list[np.ndarray]) -> list[int]:
        assigned_ids: list[int] = []
        unmatched_tracks = set(range(len(self.tracks)))

        for bbox in bboxes_xyxy:
            best_iou = 0.0
            best_idx = None
            for idx, track in enumerate(self.tracks):
                iou = bbox_iou(track.bbox_xyxy, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None and best_iou >= self.iou_threshold:
                self.tracks[best_idx].bbox_xyxy = bbox
                self.tracks[best_idx].age = 0
                unmatched_tracks.discard(best_idx)
                assigned_ids.append(self.tracks[best_idx].track_id)
            else:
                track = Track(track_id=self.next_id, bbox_xyxy=bbox)
                self.tracks.append(track)
                assigned_ids.append(track.track_id)
                self.next_id += 1

        survivors: list[Track] = []
        for idx, track in enumerate(self.tracks):
            if idx in unmatched_tracks:
                track.age += 1
            if track.age <= self.max_age:
                survivors.append(track)
        self.tracks = survivors
        return assigned_ids
