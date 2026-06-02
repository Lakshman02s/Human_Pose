from __future__ import annotations

import time

import numpy as np
from mmengine.registry import init_default_scope

from models.base import Detection


class HumanDetector:
    def __init__(self, config: str, checkpoint: str, device: str = "cpu", score_thr: float = 0.5):
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.score_thr = score_thr

        from mmdet.utils import register_all_modules
        from mmdet.apis import init_detector

        register_all_modules(init_default_scope=False)
        init_default_scope("mmdet")
        self.model = init_detector(config, checkpoint, device=device)

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[Detection], float]:
        from mmdet.apis import inference_detector

        init_default_scope("mmdet")
        start = time.perf_counter()
        result = inference_detector(self.model, frame_bgr)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        pred = result.pred_instances
        bboxes = pred.bboxes.detach().cpu().numpy() if hasattr(pred.bboxes, "detach") else np.asarray(pred.bboxes)
        scores = pred.scores.detach().cpu().numpy() if hasattr(pred.scores, "detach") else np.asarray(pred.scores)
        labels = pred.labels.detach().cpu().numpy() if hasattr(pred.labels, "detach") else np.asarray(pred.labels)

        detections: list[Detection] = []
        for bbox, score, label in zip(bboxes, scores, labels):
            if int(label) != 0 or float(score) < self.score_thr:
                continue
            detections.append(Detection(bbox_xyxy=np.asarray(bbox, dtype=np.float32), score=float(score), label=int(label)))
        return detections, elapsed_ms
