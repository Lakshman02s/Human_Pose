from __future__ import annotations

import argparse

import cv2
import numpy as np

from models.human_detector import HumanDetector
from models.model_registry import DETECTOR_MODEL, POSE_MODELS
from models.pipeline import PipelineOptions, PosePipeline, poses_to_json_serializable
from models.pose_estimator import PoseEstimator
from utils.common import ensure_dir, save_json
from utils.device import select_device
from utils.io import VideoSink, VideoSource
from utils.logging_utils import get_logger
from utils.metrics import FPSMeter, RollingStats, sample_system_usage


def _format_pose_model_label(model_name: str) -> str:
    labels = {
        "rtmpose": "RTMPose",
        "vitpose": "ViTPose",
    }
    return labels.get(model_name, model_name)


def _format_source_label(source_type: str) -> str:
    labels = {
        "video": "Video File",
        "cctv": "CCTV Footage",
        "webcam": "Webcam",
        "rtsp": "RTSP Stream",
    }
    return labels.get(source_type, source_type)


def _resize_frame_keep_aspect(frame_bgr: np.ndarray, resize_max_side: int) -> np.ndarray:
    if resize_max_side <= 0:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    max_side = max(height, width)
    if max_side <= resize_max_side:
        return frame_bgr
    scale = resize_max_side / float(max_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production-style multi-source human pose detection")
    parser.add_argument("--source", required=True, help="Webcam index, video path, or RTSP URL")
    parser.add_argument("--source-type", default="video", choices=["webcam", "video", "rtsp", "cctv"])
    parser.add_argument("--pose-model", default="rtmpose", choices=list(POSE_MODELS.keys()))
    parser.add_argument("--det-config", default=DETECTOR_MODEL.default_config)
    parser.add_argument("--det-ckpt", default=DETECTOR_MODEL.default_checkpoint)
    parser.add_argument("--pose-config", default=None)
    parser.add_argument("--pose-ckpt", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--det-score-thr", type=float, default=0.5)
    parser.add_argument("--kpt-score-thr", type=float, default=0.35)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output-video", default="outputs/pose_output.mp4")
    parser.add_argument("--output-json", default="outputs/pose_output.json")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--resize-max-side", type=int, default=0)
    parser.add_argument("--flip-test", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--disable-tracking", action="store_true")
    parser.add_argument("--disable-smoothing", action="store_true")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--rtsp-mode", default="smooth", choices=["smooth", "live"])
    parser.add_argument("--rtsp-queue-size", type=int, default=512)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logger = get_logger()

    display = args.display or not args.no_display
    device_info = select_device(args.device)
    logger.info("Device selected: %s | CUDA=%s | GPU=%s", device_info.device, device_info.cuda_available, device_info.gpu_name)
    if args.fp16 and not device_info.fp16_supported:
        logger.warning("FP16 requested but CUDA is not available. Continuing on full precision.")

    pose_spec = POSE_MODELS[args.pose_model]
    pose_config = args.pose_config or pose_spec.default_config
    pose_ckpt = args.pose_ckpt or pose_spec.default_checkpoint

    source = VideoSource(
        args.source if args.source_type != "webcam" else str(args.source),
        source_type=args.source_type,
        rtsp_mode=args.rtsp_mode,
        rtsp_queue_size=args.rtsp_queue_size,
    )
    detector = HumanDetector(args.det_config, args.det_ckpt, device=device_info.device, score_thr=args.det_score_thr)
    estimator = PoseEstimator(
        pose_config,
        pose_ckpt,
        device=device_info.device,
        flip_test=args.flip_test,
    )
    pipeline = PosePipeline(
        detector=detector,
        estimator=estimator,
        options=PipelineOptions(
            keypoint_score_thr=args.kpt_score_thr,
            enable_tracking=not args.disable_tracking,
            enable_smoothing=not args.disable_smoothing,
        ),
    )

    sink = None
    fps_meter = FPSMeter(window_size=30)
    latency_stats = RollingStats()
    frame_records = []
    frames_seen = 0
    interrupted = False
    pose_model_label = _format_pose_model_label(args.pose_model)
    device_label = "GPU" if device_info.cuda_available else "CPU"
    source_label = _format_source_label(args.source_type)

    try:
        while True:
            packet = source.read()
            if packet is None:
                break
            frame_for_inference = _resize_frame_keep_aspect(packet.frame_bgr, args.resize_max_side)
            fps = fps_meter.update()
            overlay_lines = [
                f"Pose Model: {pose_model_label}",
                f"Detector: RTMDet",
                f"Device: {device_label}",
                f"Input: {source_label}",
            ]
            result = pipeline.process_frame(
                frame_for_inference,
                frame_id=packet.frame_id,
                fps=fps,
                overlay_lines=overlay_lines,
            )
            latency_stats.add(result.total_ms)
            usage = sample_system_usage(device_info.device)

            rendered = result.rendered_frame
            if sink is None:
                sink = VideoSink(args.output_video, fps=source.fps(), size=(rendered.shape[1], rendered.shape[0]))
            sink.write(rendered)
            frame_records.append(
                {
                    "frame_id": packet.frame_id,
                    "timestamp_ms": packet.timestamp_ms,
                    "fps": fps,
                    "latency_ms": result.total_ms,
                    "detector_ms": result.detector_ms,
                    "pose_ms": result.pose_ms,
                    "cpu_percent": usage.cpu_percent,
                    "ram_percent": usage.ram_percent,
                    "gpu_memory_mb": usage.gpu_memory_mb,
                    "frame_width": int(rendered.shape[1]),
                    "frame_height": int(rendered.shape[0]),
                    "poses": poses_to_json_serializable(result.poses),
                }
            )

            if display:
                cv2.imshow("Pose Detection", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frames_seen += 1
            if args.max_frames > 0 and frames_seen >= args.max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Interrupted by user. Saving partial outputs.")
    finally:
        source.release()
        if sink is not None:
            sink.release()
        if display:
            cv2.destroyAllWindows()

    payload = {
        "run_summary": {
            "frames_processed": frames_seen,
            "latency": latency_stats.summary(),
            "source_type": args.source_type,
            "pose_model": args.pose_model,
            "run_label": args.run_label,
            "device": device_info.device,
            "rtsp_mode": args.rtsp_mode if args.source_type == "rtsp" else "",
            "resize_max_side": args.resize_max_side,
            "flip_test": args.flip_test,
            "interrupted": interrupted,
        },
        "frames": frame_records,
    }
    save_json(payload, args.output_json)
    logger.info("Saved video to %s", args.output_video)
    logger.info("Saved JSON to %s", args.output_json)


if __name__ == "__main__":
    main()
