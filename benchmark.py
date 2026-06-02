from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2

from models.human_detector import HumanDetector
from models.pipeline import PipelineOptions, PosePipeline
from models.pose_estimator import PoseEstimator
from utils.common import ensure_dir, save_json
from utils.device import select_device
from utils.io import VideoSource
from utils.logging_utils import get_logger
from utils.metrics import FPSMeter, RollingStats, sample_system_usage

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
    parser = argparse.ArgumentParser(description="Benchmark RTMPose and ViTPose across CCTV-style sources")
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-type", default="auto", choices=["auto", "video", "rtsp", "cctv", "webcam"])
    parser.add_argument("--models", nargs="+", default=["rtmpose", "vitpose"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--output-dir", default="benchmarks")
    parser.add_argument("--resize-max-side", type=int, default=0)
    parser.add_argument("--rtsp-mode", default="smooth", choices=["smooth", "live"])
    parser.add_argument("--rtsp-queue-size", type=int, default=512)
    return parser.parse_args()


def benchmark_model(
    source_path: str,
    source_type: str,
    model_name: str,
    device: str,
    max_frames: int,
    warmup_frames: int,
    resize_max_side: int,
    rtsp_mode: str,
    rtsp_queue_size: int,
    screenshot_dir: Path | None = None,
) -> dict:
    device_info = select_device(device)
    if model_name not in POSE_CONFIGS:
        raise ValueError(f"Unsupported model for benchmark: {model_name}")
    det = HumanDetector(str(DET_CONFIG), str(DET_CKPT), device=device_info.device)
    pose = PoseEstimator(str(POSE_CONFIGS[model_name]), str(POSE_CKPTS[model_name]), device=device_info.device, flip_test=False)
    pipeline = PosePipeline(det, pose, PipelineOptions())
    inferred_source_type = source_type
    if inferred_source_type == "auto":
        inferred_source_type = "rtsp" if str(source_path).startswith("rtsp://") else "video"
    source = VideoSource(
        source_path,
        source_type=inferred_source_type,
        rtsp_mode=rtsp_mode,
        rtsp_queue_size=rtsp_queue_size,
    )
    fps_meter = FPSMeter()
    latency_stats = RollingStats()
    detector_stats = RollingStats()
    pose_stats = RollingStats()
    fps_samples: list[float] = []
    usage_samples = []
    processed = 0
    screenshot_path = None

    try:
        while processed < max_frames:
            packet = source.read()
            if packet is None:
                break
            frame_bgr = packet.frame_bgr
            if resize_max_side > 0:
                height, width = frame_bgr.shape[:2]
                max_side = max(height, width)
                if max_side > resize_max_side:
                    scale = resize_max_side / float(max_side)
                    new_width = max(1, int(round(width * scale)))
                    new_height = max(1, int(round(height * scale)))
                    frame_bgr = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
            result = pipeline.process_frame(frame_bgr, packet.frame_id)
            if screenshot_dir is not None and screenshot_path is None and result.rendered_frame is not None:
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = screenshot_dir / f"{model_name}_sample_frame.jpg"
                cv2.imwrite(str(screenshot_path), result.rendered_frame)
            fps = fps_meter.update()
            if processed >= warmup_frames:
                latency_stats.add(result.total_ms)
                detector_stats.add(result.detector_ms)
                pose_stats.add(result.pose_ms)
                fps_samples.append(fps)
                usage_samples.append(sample_system_usage(device_info.device))
            processed += 1
    finally:
        source.release()

    avg_gpu_mem = 0.0
    avg_cpu = 0.0
    avg_ram = 0.0
    if usage_samples:
        avg_gpu_mem = sum(s.gpu_memory_mb for s in usage_samples) / len(usage_samples)
        avg_cpu = sum(s.cpu_percent for s in usage_samples) / len(usage_samples)
        avg_ram = sum(s.ram_percent for s in usage_samples) / len(usage_samples)

    return {
        "model": model_name,
        "device": device_info.device,
        "frames_processed": processed,
        "fps_mean": sum(fps_samples) / len(fps_samples) if fps_samples else 0.0,
        "latency_ms_mean": latency_stats.summary()["mean"],
        "latency_ms_p95": latency_stats.summary()["p95"],
        "detector_ms_mean": detector_stats.summary()["mean"],
        "pose_ms_mean": pose_stats.summary()["mean"],
        "gpu_memory_mb_mean": avg_gpu_mem,
        "cpu_percent_mean": avg_cpu,
        "ram_percent_mean": avg_ram,
        "resize_max_side": resize_max_side,
        "sample_frame": str(screenshot_path) if screenshot_path else "",
    }


def main() -> None:
    args = parse_args()
    logger = get_logger("benchmark")
    output_dir = ensure_dir(args.output_dir)
    history_dir = ensure_dir(output_dir / "history")
    rows = []
    for model_name in args.models:
        logger.info("Benchmarking %s", model_name)
        rows.append(
            benchmark_model(
                args.source,
                args.source_type,
                model_name,
                args.device,
                args.max_frames,
                args.warmup_frames,
                args.resize_max_side,
                args.rtsp_mode,
                args.rtsp_queue_size,
                screenshot_dir=output_dir,
            )
        )

    json_path = output_dir / "benchmark_summary.json"
    csv_path = output_dir / "benchmark_summary.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_json_path = history_dir / f"benchmark_{timestamp}.json"
    history_csv_path = history_dir / f"benchmark_{timestamp}.csv"
    payload = {
        "timestamp": timestamp,
        "source": args.source,
        "source_type": args.source_type,
        "device": args.device,
        "max_frames": args.max_frames,
        "warmup_frames": args.warmup_frames,
        "resize_max_side": args.resize_max_side,
        "rtsp_mode": args.rtsp_mode if args.source_type == "rtsp" else "",
        "results": rows,
    }
    save_json(rows, json_path)
    save_json(payload, history_json_path)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with history_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved benchmark JSON to %s", json_path)
    logger.info("Saved benchmark CSV to %s", csv_path)
    logger.info("Saved benchmark history JSON to %s", history_json_path)
    logger.info("Saved benchmark history CSV to %s", history_csv_path)


if __name__ == "__main__":
    main()
