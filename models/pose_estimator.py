from __future__ import annotations

import time

import numpy as np
from mmengine import Config
from mmengine.registry import init_default_scope

from models.base import PoseResult


class PoseEstimator:
    def __init__(self, config: str, checkpoint: str, device: str = "cpu", flip_test: bool = False):
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.flip_test = flip_test

        from mmpose.utils import register_all_modules
        from mmpose.apis import init_model

        register_all_modules(init_default_scope=False)
        init_default_scope("mmpose")
        cfg = Config.fromfile(config)
        if getattr(cfg.model, "test_cfg", None) is None:
            cfg.model.test_cfg = {}
        cfg.model.test_cfg["flip_test"] = flip_test
        self.model = init_model(cfg, checkpoint, device=device)

    def predict(self, frame_bgr: np.ndarray, bboxes_xyxy: list[np.ndarray]) -> tuple[list[PoseResult], float]:
        if not bboxes_xyxy:
            return [], 0.0

        from mmpose.apis import inference_topdown

        init_default_scope("mmpose")
        start = time.perf_counter()
        results = inference_topdown(self.model, frame_bgr, bboxes_xyxy, bbox_format="xyxy")
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        poses: list[PoseResult] = []
        for sample, bbox in zip(results, bboxes_xyxy):
            pred = sample.pred_instances
            keypoints = pred.keypoints[0]
            if hasattr(keypoints, "detach"):
                keypoints = keypoints.detach().cpu().numpy()
            keypoint_scores = pred.keypoint_scores[0]
            if hasattr(keypoint_scores, "detach"):
                keypoint_scores = keypoint_scores.detach().cpu().numpy()
            keypoints_xyc = np.concatenate([keypoints[:, :2], keypoint_scores[:, None]], axis=1).astype(np.float32)
            bbox_score = float(np.mean(keypoint_scores))
            poses.append(PoseResult(bbox_xyxy=np.asarray(bbox, dtype=np.float32), score=bbox_score, keypoints_xyc=keypoints_xyc))
        return poses, elapsed_ms
