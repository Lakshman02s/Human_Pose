from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from models.pipeline import PosePipeline
from utils.common import save_json


class COCOKeypointEvaluator:
    def __init__(self, annotation_file: str | Path, image_root: str | Path):
        self.annotation_file = str(annotation_file)
        self.image_root = Path(image_root)
        self.coco_gt = COCO(self.annotation_file)

    def run(self, pipeline: PosePipeline, output_json: str | Path, max_images: int | None = None) -> dict:
        image_ids = self.coco_gt.getImgIds()
        if max_images is not None:
            image_ids = image_ids[:max_images]

        predictions = []
        image_person_counts = defaultdict(int)

        for image_id in tqdm(image_ids, desc="Evaluating COCO"):
            img_info = self.coco_gt.loadImgs([image_id])[0]
            img_path = self.image_root / img_info["file_name"]
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            result = pipeline.process_frame(frame, frame_id=image_id)
            for pose in result.poses:
                keypoints = []
                visible = 0
                for kp in pose.keypoints_xyc:
                    visibility = 2 if kp[2] > 0.05 else 0
                    visible += int(visibility > 0)
                    keypoints.extend([float(kp[0]), float(kp[1]), visibility])
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": 1,
                        "keypoints": keypoints,
                        "score": float(pose.score),
                        "num_keypoints": visible,
                    }
                )
                image_person_counts[image_id] += 1

        save_json(predictions, output_json)
        coco_dt = self.coco_gt.loadRes(str(output_json))
        coco_eval = COCOeval(self.coco_gt, coco_dt, "keypoints")
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        summary = {
            "mAP": float(coco_eval.stats[0]),
            "mAP50": float(coco_eval.stats[1]),
            "mAP75": float(coco_eval.stats[2]),
            "mAR": float(coco_eval.stats[5]),
            "precision": float(coco_eval.stats[1]),
            "recall": float(coco_eval.stats[5]),
            "images_evaluated": len(image_ids),
            "predictions": len(predictions),
        }
        return summary
