from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean

import cv2
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from generate_report import (
    build_executive_summary,
    build_findings,
    build_recommendations,
    load_run_summaries,
)

ROOT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = ROOT_DIR / "benchmarks"
BENCHMARK_HISTORY_DIR = BENCHMARKS_DIR / "history"
OUTPUTS_DIR = ROOT_DIR / "outputs"
VIDEOS_DIR = ROOT_DIR / "videos"
CONFIGS_DIR = ROOT_DIR / "configs"
DASHBOARD_STATIC_DIR = ROOT_DIR / "dashboard" / "static"
DASHBOARD_CACHE_DIR = ROOT_DIR / "dashboard" / "cache"
DASHBOARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Pose Benchmark Dashboard", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC)) if cap.isOpened() else 0
    codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip() or "unknown"
    meta = {
        "path": str(path.relative_to(ROOT_DIR)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": codec,
    }
    cap.release()
    return meta


def _ensure_web_video(path: Path) -> Path | None:
    if not path.exists():
        return None

    DASHBOARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    web_path = DASHBOARD_CACHE_DIR / f"{path.stem}.web.mp4"
    if web_path.exists() and web_path.stat().st_mtime >= path.stat().st_mtime:
        return web_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(web_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return web_path if web_path.exists() else None


def _ensure_thumbnail(path: Path) -> Path | None:
    if not path.exists():
        return None

    DASHBOARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = DASHBOARD_CACHE_DIR / f"{path.stem}.thumb.jpg"
    if thumb_path.exists() and thumb_path.stat().st_mtime >= path.stat().st_mtime:
        return thumb_path

    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    resized = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    if cv2.imwrite(str(thumb_path), resized):
        return thumb_path
    return None


def _infer_run_name(json_path: Path, run_summary: dict, model: str) -> tuple[str, Path | None]:
    run_label = str(run_summary.get("run_label", "")).strip()
    source_type = run_summary.get("source_type", "video")
    stem = json_path.stem

    if source_type == "rtsp":
        if run_label:
            return run_label, None
        match = re.match(r"(.+?)_(rtmpose|vitpose)_\d{8}_\d{6}$", stem)
        return (match.group(1) if match else stem, None)

    if run_label:
        video_name = run_label if run_label.endswith(".mp4") else f"{run_label}.mp4"
        video_path = VIDEOS_DIR / video_name
        return video_name, video_path if video_path.exists() else None

    if model != "unknown" and stem.endswith(f"_{model}_output"):
        video_name = stem[: -len(f"_{model}_output")]
        video_path = VIDEOS_DIR / f"{video_name}.mp4"
        return f"{video_name}.mp4", video_path if video_path.exists() else None

    return stem, None


def _build_selection_info(json_path: Path, run_summary: dict, model: str, display_name: str) -> tuple[str, str]:
    source_type = run_summary.get("source_type", "video")
    stem = json_path.stem
    if source_type == "rtsp":
        match = re.match(r"(.+?)_(rtmpose|vitpose)_(\d{8}_\d{6})$", stem)
        if match:
            camera_name, _, timestamp = match.groups()
            try:
                formatted_ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                formatted_ts = timestamp
            return stem, f"{camera_name} | {formatted_ts} | {model.upper()}"
        return stem, f"{display_name} | {model.upper()}"
    return display_name, display_name


def _derive_failure_metrics(frames: list[dict]) -> dict:
    zero_people = 0
    low_confidence = 0
    crowded_failures = 0
    distance_failures = 0
    occlusion_failures = 0
    angle_failures = 0
    confidences = []

    for frame in frames:
        poses = frame.get("poses", [])
        if not poses:
            zero_people += 1
            distance_failures += 1
            continue

        pose_scores = [pose.get("score", 0.0) for pose in poses]
        frame_conf = fmean(pose_scores) if pose_scores else 0.0
        confidences.append(frame_conf)
        if frame_conf < 0.45:
            low_confidence += 1
            occlusion_failures += 1
        if len(poses) >= 4 and frame_conf < 0.6:
            crowded_failures += 1
        if len(poses) <= 1 and frame_conf < 0.5:
            angle_failures += 1

    return {
        "zero_people_frames": zero_people,
        "low_confidence_frames": low_confidence,
        "crowding_failures": crowded_failures,
        "distance_failures": distance_failures,
        "occlusion_failures": occlusion_failures,
        "angle_failures": angle_failures,
        "avg_keypoint_confidence": round(fmean(confidences), 4) if confidences else 0.0,
    }


def _classify_stream_quality(metrics: dict, video_meta: dict | None, source_type: str) -> str:
    if source_type != "rtsp":
        return "stable"
    fps = float(metrics.get("avg_fps", 0.0))
    latency = float(metrics.get("avg_latency_ms", 0.0))
    source_fps = float((video_meta or {}).get("fps") or 0.0)
    if source_fps > 0 and fps < source_fps * 0.45:
        return "degraded"
    if latency > 220:
        return "degraded"
    if latency > 400 or fps < 5:
        return "critical"
    return "stable"


def _derive_stream_metrics(metrics: dict, video_meta: dict | None, run_summary: dict) -> dict:
    source_fps = float((video_meta or {}).get("fps") or 0.0)
    processed_fps = float(metrics.get("avg_fps", 0.0))
    dropped_ratio = 0.0
    if source_fps > 0:
        dropped_ratio = max(0.0, 1.0 - min(processed_fps / source_fps, 1.0))
    jitter_ms = float(run_summary.get("latency", {}).get("p95", 0.0)) - float(run_summary.get("latency", {}).get("p50", 0.0))
    return {
        "stream_fps": round(source_fps, 2),
        "processed_fps": round(processed_fps, 2),
        "dropped_frame_ratio": round(dropped_ratio, 3),
        "stream_latency_ms": round(float(metrics.get("avg_latency_ms", 0.0)), 2),
        "network_jitter_ms": round(max(jitter_ms, 0.0), 2),
        "codec": (video_meta or {}).get("codec", "unknown"),
        "reconnection_count": 0,
    }


def _load_run_artifacts() -> list[dict]:
    runs = []
    for json_path in sorted(OUTPUTS_DIR.glob("*.json")):
        data = _read_json(json_path)
        if "run_summary" not in data or "frames" not in data:
            continue
        frames = data.get("frames", [])
        run_summary = data.get("run_summary", {})
        model = run_summary.get("pose_model", "unknown")
        stem = json_path.stem
        display_name, video_path = _infer_run_name(json_path, run_summary, model)
        selection_key, selection_label = _build_selection_info(json_path, run_summary, model, display_name)
        output_video = OUTPUTS_DIR / f"{stem}.mp4"
        web_video = _ensure_web_video(output_video) if output_video.exists() else None
        thumbnail = _ensure_thumbnail(output_video) if output_video.exists() else None

        fps_values = [frame.get("fps", 0.0) for frame in frames if isinstance(frame.get("fps", 0.0), (int, float))]
        latency_values = [frame.get("latency_ms", 0.0) for frame in frames if isinstance(frame.get("latency_ms", 0.0), (int, float))]
        cpu_values = [frame.get("cpu_percent", 0.0) for frame in frames]
        ram_values = [frame.get("ram_percent", 0.0) for frame in frames]
        gpu_values = [frame.get("gpu_memory_mb", 0.0) for frame in frames]
        people_counts = [len(frame.get("poses", [])) for frame in frames]
        video_meta = _video_meta(video_path) if video_path and video_path.exists() else (_video_meta(output_video) if output_video.exists() else None)
        metrics = {
            "avg_fps": round(fmean(fps_values), 3) if fps_values else 0.0,
            "avg_latency_ms": round(fmean(latency_values), 3) if latency_values else 0.0,
            "avg_cpu_percent": round(fmean(cpu_values), 3) if cpu_values else 0.0,
            "avg_ram_percent": round(fmean(ram_values), 3) if ram_values else 0.0,
            "avg_gpu_memory_mb": round(fmean(gpu_values), 3) if gpu_values else 0.0,
            "avg_people_count": round(fmean(people_counts), 3) if people_counts else 0.0,
        }
        stream_metrics = _derive_stream_metrics(metrics, video_meta, run_summary)
        stream_quality = _classify_stream_quality(metrics, video_meta, run_summary.get("source_type", "video"))

        runs.append(
            {
                "id": stem,
                "model": model,
                "video_name": display_name,
                "selection_key": selection_key,
                "selection_label": selection_label,
                "created_at": datetime.fromtimestamp(json_path.stat().st_mtime).isoformat(timespec="seconds"),
                "video_meta": video_meta,
                "output_video": str(output_video.relative_to(ROOT_DIR)) if output_video.exists() else None,
                "output_video_web": str(web_video.relative_to(ROOT_DIR)) if web_video else None,
                "thumbnail": str(thumbnail.relative_to(ROOT_DIR)) if thumbnail else None,
                "run_summary": run_summary,
                "failure_metrics": _derive_failure_metrics(frames),
                "metrics": metrics,
                "stream_metrics": stream_metrics,
                "stream_quality": stream_quality,
                "series": {
                    "fps": fps_values[:400],
                    "latency_ms": latency_values[:400],
                    "cpu_percent": cpu_values[:400],
                    "ram_percent": ram_values[:400],
                    "gpu_memory_mb": gpu_values[:400],
                    "people_count": people_counts[:400],
                },
            }
        )
    return runs


def _generate_recommendations(benchmark_df: pd.DataFrame, run_df: pd.DataFrame) -> list[dict]:
    recommendations: list[dict] = []
    if benchmark_df.empty:
        return recommendations

    ranked = benchmark_df.sort_values(["latency_ms_mean", "fps_mean"], ascending=[True, False]).reset_index(drop=True)
    recommended = ranked.iloc[0]
    recommendations.append(
        {
            "title": f"{str(recommended['model']).upper()} recommended for operational deployment",
            "summary": (
                f"On the current setup it delivered the strongest balance of throughput and latency "
                f"({recommended['fps_mean']:.2f} FPS, {recommended['latency_ms_mean']:.2f} ms)."
            ),
            "severity": "positive",
        }
    )
    slower = ranked.iloc[-1]
    if str(slower["model"]).lower() != str(recommended["model"]).lower():
        recommendations.append(
            {
                "title": f"{str(slower['model']).upper()} is better suited for offline accuracy review",
                "summary": (
                    f"It is materially slower on the current CPU-only benchmark "
                    f"({slower['fps_mean']:.2f} FPS, {slower['latency_ms_mean']:.2f} ms), so it fits comparison workloads better than real-time deployment."
                ),
                "severity": "neutral",
            }
        )

    if not run_df.empty:
        if (run_df["zero_people_ratio"] > 0.5).any():
            recommendations.append(
                {
                    "title": "Crowded and far-distance scenes need stronger detection support",
                    "summary": "Several real-world runs showed high no-person ratios, indicating the detector misses small or distant subjects before pose estimation can start.",
                    "severity": "warning",
                }
            )
        if (run_df["source_type"] == "rtsp").any():
            recommendations.append(
                {
                    "title": "RTSP stream quality should be monitored alongside pose performance",
                    "summary": "Live feed stability depends on codec behavior, stream transport quality, and processing speed, so stream health metrics should remain part of deployment readiness decisions.",
                    "severity": "warning",
                }
            )
    return recommendations


def _build_platform_payload() -> dict:
    latest_path = BENCHMARKS_DIR / "benchmark_summary.json"
    latest_benchmark = _read_json(latest_path) if latest_path.exists() else []
    benchmark_csv = BENCHMARKS_DIR / "benchmark_summary.csv"
    benchmark_df = pd.read_csv(benchmark_csv) if benchmark_csv.exists() else pd.DataFrame()
    runs = _load_run_artifacts()
    history = _load_benchmark_history()
    official = _load_official_coco_baselines()
    run_df = load_run_summaries()

    overview = {
        "recommended_model": (
            str(benchmark_df.sort_values(["latency_ms_mean", "fps_mean"], ascending=[True, False]).iloc[0]["model"]).upper()
            if not benchmark_df.empty
            else "RTMPOSE"
        ),
        "total_runs": len(runs),
        "recorded_runs": sum(1 for run in runs if run["run_summary"].get("source_type") == "video"),
        "live_runs": sum(1 for run in runs if run["run_summary"].get("source_type") == "rtsp"),
        "avg_fps": round(fmean([run["metrics"]["avg_fps"] for run in runs]) if runs else 0.0, 2),
        "avg_latency_ms": round(fmean([run["metrics"]["avg_latency_ms"] for run in runs]) if runs else 0.0, 2),
        "avg_confidence": round(fmean([run["failure_metrics"]["avg_keypoint_confidence"] for run in runs]) if runs else 0.0, 3),
        "system_status": "Stable" if runs else "No Data",
        "live_status": "Validated" if any(run["run_summary"].get("source_type") == "rtsp" for run in runs) else "Pending",
    }

    model_profiles = []
    if not run_df.empty:
        for model, model_df in run_df.groupby("model"):
            benchmark_row = benchmark_df[benchmark_df["model"] == model]
            official_row = next((row for row in official.get("models", []) if row.get("model") == model), None)
            success_ratio = float((model_df["zero_people_ratio"] < 0.3).mean()) if len(model_df) else 0.0
            model_profiles.append(
                {
                    "model": model,
                    "runs": int(len(model_df)),
                    "avg_fps": round(float(model_df["avg_fps"].mean()), 2),
                    "avg_latency_ms": round(float(model_df["avg_latency_ms"].mean()), 2),
                    "avg_cpu_percent": round(float(model_df.get("avg_cpu_percent", pd.Series([0] * len(model_df))).mean()), 2) if "avg_cpu_percent" in model_df else round(float(fmean([run["metrics"]["avg_cpu_percent"] for run in runs if run["model"] == model])), 2),
                    "avg_ram_percent": round(float(fmean([run["metrics"]["avg_ram_percent"] for run in runs if run["model"] == model])), 2),
                    "avg_confidence": round(float(model_df["avg_keypoint_confidence"].mean()), 3),
                    "zero_people_ratio": round(float(model_df["zero_people_ratio"].mean()), 3),
                    "success_ratio": round(success_ratio, 3),
                    "official_map": round(float(official_row.get("mAP", 0.0)), 3) if official_row else 0.0,
                    "benchmark_fps": round(float(benchmark_row["fps_mean"].iloc[0]), 2) if not benchmark_row.empty else 0.0,
                    "benchmark_latency_ms": round(float(benchmark_row["latency_ms_mean"].iloc[0]), 2) if not benchmark_row.empty else 0.0,
                }
            )

    timeline = [
        {
            "timestamp": entry.get("timestamp", ""),
            "source": entry.get("source", ""),
            "device": entry.get("device", ""),
            "models": entry.get("models", []),
            "max_frames": entry.get("max_frames", 0),
        }
        for entry in history
    ]

    distributions = {
        "latency": {model: [] for model in ["rtmpose", "vitpose"]},
        "fps": {model: [] for model in ["rtmpose", "vitpose"]},
        "confidence": {model: [] for model in ["rtmpose", "vitpose"]},
    }
    for run in runs:
        model = run["model"]
        distributions["latency"].setdefault(model, []).extend(run["series"]["latency_ms"][:120])
        distributions["fps"].setdefault(model, []).extend(run["series"]["fps"][:120])
        distributions["confidence"].setdefault(model, []).append(run["failure_metrics"]["avg_keypoint_confidence"])

    failure_rows = []
    for run in runs:
        failure_rows.append(
            {
                "selection_key": run["selection_key"],
                "label": run["selection_label"],
                "model": run["model"],
                "source_type": run["run_summary"].get("source_type", "video"),
                "distance": run["failure_metrics"]["distance_failures"],
                "occlusion": run["failure_metrics"]["occlusion_failures"],
                "crowding": run["failure_metrics"]["crowding_failures"],
                "angle": run["failure_metrics"]["angle_failures"],
                "severity": (
                    "critical"
                    if run["failure_metrics"]["distance_failures"] > 100 or run["failure_metrics"]["occlusion_failures"] > 80
                    else "warning"
                    if run["failure_metrics"]["crowding_failures"] > 30
                    else "stable"
                ),
            }
        )

    video_validations = [run for run in runs if run["run_summary"].get("source_type") == "video"]
    live_runs = [run for run in runs if run["run_summary"].get("source_type") == "rtsp"]

    rtsp_aggregate = {
        "healthy_streams": sum(1 for run in live_runs if run["stream_quality"] == "stable"),
        "degraded_streams": sum(1 for run in live_runs if run["stream_quality"] == "degraded"),
        "critical_streams": sum(1 for run in live_runs if run["stream_quality"] == "critical"),
        "avg_stream_latency_ms": round(fmean([run["stream_metrics"]["stream_latency_ms"] for run in live_runs]) if live_runs else 0.0, 2),
        "avg_dropped_ratio": round(fmean([run["stream_metrics"]["dropped_frame_ratio"] for run in live_runs]) if live_runs else 0.0, 3),
    }

    usage_distribution = defaultdict(int)
    for run in runs:
        usage_distribution[run["model"]] += 1

    return {
        "overview": overview,
        "latest_benchmark": latest_benchmark,
        "model_profiles": model_profiles,
        "official_coco_baselines": official,
        "recommendations": _generate_recommendations(benchmark_df, run_df),
        "runs": runs,
        "video_validations": video_validations,
        "live_runs": live_runs,
        "rtsp_aggregate": rtsp_aggregate,
        "benchmark_history": history,
        "timeline": timeline,
        "distributions": distributions,
        "failure_rows": failure_rows,
        "usage_distribution": dict(usage_distribution),
        "coco_summaries": _load_coco_summaries(),
    }


def _load_benchmark_history() -> list[dict]:
    history = []
    for path in sorted(BENCHMARK_HISTORY_DIR.glob("benchmark_*.json")):
        payload = _read_json(path)
        history.append(payload)
    return history


def _load_coco_summaries() -> list[dict]:
    summaries = []
    for path in sorted(OUTPUTS_DIR.glob("coco_eval_summary*.json")):
        payload = _read_json(path)
        model = "unknown"
        stem = path.stem.lower()
        if "rtmpose" in stem:
            model = "rtmpose"
        elif "vitpose" in stem:
            model = "vitpose"
        elif isinstance(payload, dict):
            model = str(payload.get("pose_model", "unknown")).lower()
        summaries.append(
            {
                "file": path.name,
                "model": model,
                "mAP": float(payload.get("mAP", 0.0)),
                "mAP50": float(payload.get("mAP50", 0.0)),
                "mAP75": float(payload.get("mAP75", 0.0)),
                "mAR": float(payload.get("mAR", 0.0)),
                "precision": float(payload.get("precision", 0.0)),
                "recall": float(payload.get("recall", 0.0)),
                "images_evaluated": int(payload.get("images_evaluated", 0)),
                "predictions": int(payload.get("predictions", 0)),
            }
        )
    return summaries


def _load_official_coco_baselines() -> dict:
    path = CONFIGS_DIR / "official_coco_baselines.json"
    if not path.exists():
        return {"source": "", "notes": [], "models": []}
    return _read_json(path)


def _build_dashboard_payload() -> dict:
    latest_benchmark = []
    latest_path = BENCHMARKS_DIR / "benchmark_summary.json"
    if latest_path.exists():
        latest_benchmark = _read_json(latest_path)

    runs = _load_run_artifacts()
    history = _load_benchmark_history()
    grouped_runs = defaultdict(dict)
    for run in runs:
        grouped_runs[run["video_name"]][run["model"]] = run

    comparison_cards = []
    for video_name, model_runs in grouped_runs.items():
        if "rtmpose" in model_runs and "vitpose" in model_runs:
            r = model_runs["rtmpose"]["metrics"]
            v = model_runs["vitpose"]["metrics"]
            comparison_cards.append(
                {
                    "video_name": video_name,
                    "fps_delta": round(r["avg_fps"] - v["avg_fps"], 3),
                    "latency_delta_ms": round(v["avg_latency_ms"] - r["avg_latency_ms"], 3),
                    "people_delta": round(r["avg_people_count"] - v["avg_people_count"], 3),
                    "recommended_model": "rtmpose" if r["avg_latency_ms"] <= v["avg_latency_ms"] else "vitpose",
                }
            )

    return {
        "latest_benchmark": latest_benchmark,
        "benchmark_history": history,
        "coco_summaries": _load_coco_summaries(),
        "official_coco_baselines": _load_official_coco_baselines(),
        "runs": runs,
        "comparisons": comparison_cards,
    }


def _build_report_payload() -> dict:
    benchmark_path = BENCHMARKS_DIR / "benchmark_summary.csv"
    benchmark_df = pd.read_csv(benchmark_path) if benchmark_path.exists() else pd.DataFrame()
    official_coco = _load_official_coco_baselines()
    run_df = load_run_summaries()

    summary_lines = build_executive_summary(run_df, benchmark_df)
    findings = build_findings(run_df)
    recommendations = build_recommendations()

    recorded_df = run_df[run_df["source_type"] == "video"].copy() if not run_df.empty else pd.DataFrame()
    live_df = run_df[run_df["source_type"] == "rtsp"].copy() if not run_df.empty else pd.DataFrame()

    aggregate_df = pd.DataFrame()
    if not run_df.empty:
        aggregate_df = (
            run_df.groupby(["source_type", "model"], as_index=False)
            .agg(
                runs=("run_id", "count"),
                mean_fps=("avg_fps", "mean"),
                mean_latency_ms=("avg_latency_ms", "mean"),
                mean_people=("avg_people_count", "mean"),
                mean_keypoint_conf=("avg_keypoint_confidence", "mean"),
                mean_zero_people_ratio=("zero_people_ratio", "mean"),
            )
            .round(3)
        )

    latest_history = _load_benchmark_history()
    latest_benchmark = benchmark_df.to_dict(orient="records") if not benchmark_df.empty else []

    recommended_model = "rtmpose"
    if not benchmark_df.empty and "latency_ms_mean" in benchmark_df.columns:
        recommended_model = (
            benchmark_df.sort_values(["latency_ms_mean", "fps_mean"], ascending=[True, False]).iloc[0]["model"]
        )

    highlights = {
        "total_runs": int(len(run_df)) if not run_df.empty else 0,
        "recorded_runs": int((run_df["source_type"] == "video").sum()) if not run_df.empty else 0,
        "live_runs": int((run_df["source_type"] == "rtsp").sum()) if not run_df.empty else 0,
        "recommended_model": str(recommended_model).upper(),
        "live_camera_validated": bool(not live_df.empty),
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "official_coco_baselines": official_coco,
        "benchmark_rows": latest_benchmark,
        "run_highlights": highlights,
        "executive_summary": summary_lines,
        "findings": findings,
        "recommendations": recommendations,
        "aggregate_rows": aggregate_df.to_dict(orient="records") if not aggregate_df.empty else [],
        "recorded_rows": recorded_df.to_dict(orient="records") if not recorded_df.empty else [],
        "live_rows": live_df.to_dict(orient="records") if not live_df.empty else [],
        "history_rows": latest_history,
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/dashboard")
def dashboard() -> dict:
    return _build_platform_payload()


@app.get("/api/report")
def report() -> dict:
    payload = _build_report_payload()
    payload["platform"] = _build_platform_payload()
    return payload


@app.get("/api/runs")
def list_runs() -> list[dict]:
    return _load_run_artifacts()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    for run in _load_run_artifacts():
        if run["id"] == run_id:
            return run
    raise HTTPException(status_code=404, detail="Run not found")


@app.post("/api/runs/{run_id}/preview")
def prepare_run_preview(run_id: str) -> dict:
    for run in _load_run_artifacts():
        if run["id"] != run_id:
            continue
        output_video = run.get("output_video")
        if not output_video:
            raise HTTPException(status_code=404, detail="Output video not found")
        web_video = _ensure_web_video(ROOT_DIR / output_video)
        preview_path = web_video if web_video else ROOT_DIR / output_video
        return {
            "run_id": run_id,
            "preview_video": str(preview_path.relative_to(ROOT_DIR)),
            "is_web_optimized": web_video is not None,
        }
    raise HTTPException(status_code=404, detail="Run not found")


@app.get("/api/benchmarks/latest")
def latest_benchmark() -> list[dict]:
    path = BENCHMARKS_DIR / "benchmark_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Latest benchmark not found")
    return _read_json(path)


@app.get("/api/benchmarks/history")
def benchmark_history() -> list[dict]:
    return _load_benchmark_history()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DASHBOARD_STATIC_DIR / "index.html")


@app.get("/report")
def report_page() -> FileResponse:
    return FileResponse(DASHBOARD_STATIC_DIR / "report.html")


app.mount("/dashboard", StaticFiles(directory=DASHBOARD_STATIC_DIR), name="dashboard")
app.mount("/dashboard-cache", StaticFiles(directory=DASHBOARD_CACHE_DIR), name="dashboard-cache")
app.mount("/artifacts", StaticFiles(directory=ROOT_DIR), name="artifacts")
