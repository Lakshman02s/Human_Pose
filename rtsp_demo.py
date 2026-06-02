from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from main import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTSP pose detection demo")
    parser.add_argument("--rtsp-url", required=True)
    parser.add_argument("--pose-model", default="rtmpose", choices=["rtmpose", "vitpose"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera-name", default="vacron_live")
    parser.add_argument("--output-dir", default="outputs")
    args, unknown = parser.parse_known_args()
    script_dir = Path(__file__).resolve().parent
    det_config = script_dir / "configs" / "mmpose_configs" / "rtmdet_nano_320-8xb32_coco-person.py"
    det_ckpt = script_dir / "checkpoints" / "rtmdet_nano_person.pth"
    if args.pose_model == "rtmpose":
        pose_config = script_dir / "configs" / "mmpose_configs" / "rtmpose-m_8xb256-420e_coco-256x192.py"
        pose_ckpt = script_dir / "checkpoints" / "rtmpose_m.pth"
    else:
        pose_config = script_dir / "configs" / "mmpose_configs" / "td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py"
        pose_ckpt = script_dir / "checkpoints" / "vitpose_base.pth"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / args.output_dir
    output_video = output_dir / f"{args.camera_name}_{args.pose_model}_{timestamp}.mp4"
    output_json = output_dir / f"{args.camera_name}_{args.pose_model}_{timestamp}.json"

    sys.argv = [
        "rtsp_demo.py",
        "--source",
        args.rtsp_url,
        "--source-type",
        "rtsp",
        "--device",
        args.device,
        "--pose-model",
        args.pose_model,
        "--det-config",
        str(det_config),
        "--det-ckpt",
        str(det_ckpt),
        "--pose-config",
        str(pose_config),
        "--pose-ckpt",
        str(pose_ckpt),
        "--output-video",
        str(output_video),
        "--output-json",
        str(output_json),
        "--run-label",
        args.camera_name,
        "--display",
        *unknown,
    ]
    main()
