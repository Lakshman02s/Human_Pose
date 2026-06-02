from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.coco_evaluator import COCOKeypointEvaluator
from models.human_detector import HumanDetector
from models.pipeline import PipelineOptions, PosePipeline
from models.pose_estimator import PoseEstimator
from utils.common import save_json
from utils.device import select_device
from utils.logging_utils import get_logger

ROOT_DIR = Path(__file__).resolve().parent
DET_CONFIG = ROOT_DIR / "configs/mmpose_configs/rtmdet_nano_320-8xb32_coco-person.py"
DET_CKPT = ROOT_DIR / "checkpoints/rtmdet_nano_person.pth"
POSE_CONFIGS = {
    "rtmpose": ROOT_DIR / "configs/mmpose_configs/rtmpose-m_8xb256-420e_coco-256x192.py",
    "vitpose": ROOT_DIR / "configs/mmpose_configs/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py",
}
POSE_CKPTS = {
    "rtmpose": ROOT_DIR / "checkpoints/rtmpose_m.pth",
    "vitpose": ROOT_DIR / "checkpoints/vitpose_base.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RTMPose or ViTPose on COCO keypoints")
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--pose-model", default="rtmpose", choices=list(POSE_CONFIGS.keys()))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default="outputs/coco_predictions.json")
    parser.add_argument("--summary-json", default="outputs/coco_eval_summary.json")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("evaluate_coco")
    device_info = select_device(args.device)

    det = HumanDetector(str(DET_CONFIG), str(DET_CKPT), device=device_info.device)
    pose = PoseEstimator(str(POSE_CONFIGS[args.pose_model]), str(POSE_CKPTS[args.pose_model]), device=device_info.device)
    pipeline = PosePipeline(det, pose, PipelineOptions())

    evaluator = COCOKeypointEvaluator(args.annotation_file, args.image_root)
    summary = evaluator.run(pipeline, args.output_json, max_images=args.max_images)
    summary["pose_model"] = args.pose_model
    save_json(summary, args.summary_json)
    logger.info("Evaluation summary: %s", summary)


if __name__ == "__main__":
    main()
