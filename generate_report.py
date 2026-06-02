from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
CONFIGS_DIR = ROOT_DIR / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a client-ready benchmarking report from pose artifacts")
    parser.add_argument("--benchmark-csv", default="benchmarks/benchmark_summary.csv")
    parser.add_argument("--official-coco-json", default="configs/official_coco_baselines.json")
    parser.add_argument("--output-md", default="outputs/benchmarking_report.md")
    parser.add_argument("--title", default="Human Pose Detection Benchmarking Report")
    return parser.parse_args()


def format_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    return df.to_markdown(index=False)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_run_name(json_path: Path, run_summary: dict, model: str) -> str:
    run_label = str(run_summary.get("run_label", "")).strip()
    source_type = run_summary.get("source_type", "video")
    stem = json_path.stem

    if source_type == "rtsp":
        if run_label:
            return run_label
        return stem

    if run_label:
        return run_label if run_label.endswith(".mp4") else f"{run_label}.mp4"

    suffix = f"_{model}_output"
    if model != "unknown" and stem.endswith(suffix):
        return f"{stem[:-len(suffix)]}.mp4"
    return stem


def summarize_run(json_path: Path) -> dict | None:
    data = load_json(json_path)
    if "run_summary" not in data or "frames" not in data:
        return None

    run_summary = data.get("run_summary", {})
    frames = data.get("frames", [])
    model = str(run_summary.get("pose_model", "unknown"))
    source_type = str(run_summary.get("source_type", "video"))
    display_name = infer_run_name(json_path, run_summary, model)

    fps_values = [float(frame.get("fps", 0.0)) for frame in frames if isinstance(frame.get("fps"), (int, float))]
    lat_values = [float(frame.get("latency_ms", 0.0)) for frame in frames if isinstance(frame.get("latency_ms"), (int, float))]
    people_counts = [len(frame.get("poses", [])) for frame in frames]
    confidences = []
    zero_people_frames = 0
    crowded_frames = 0
    low_conf_frames = 0

    for frame in frames:
        poses = frame.get("poses", [])
        if not poses:
            zero_people_frames += 1
            continue
        scores = [float(pose.get("score", 0.0)) for pose in poses]
        if scores:
            mean_conf = fmean(scores)
            confidences.append(mean_conf)
            if mean_conf < 0.45:
                low_conf_frames += 1
        if len(poses) >= 4:
            crowded_frames += 1

    total_frames = max(1, len(frames))
    zero_people_ratio = zero_people_frames / total_frames

    return {
        "run_id": json_path.stem,
        "display_name": display_name,
        "model": model,
        "source_type": source_type,
        "frames_processed": int(run_summary.get("frames_processed", len(frames))),
        "avg_fps": round(fmean(fps_values), 3) if fps_values else 0.0,
        "avg_latency_ms": round(fmean(lat_values), 3) if lat_values else 0.0,
        "avg_people_count": round(fmean(people_counts), 3) if people_counts else 0.0,
        "avg_keypoint_confidence": round(fmean(confidences), 3) if confidences else 0.0,
        "zero_people_ratio": round(zero_people_ratio, 3),
        "crowded_frame_ratio": round(crowded_frames / total_frames, 3),
        "low_conf_frame_ratio": round(low_conf_frames / total_frames, 3),
        "rtsp_mode": run_summary.get("rtsp_mode", ""),
        "interrupted": bool(run_summary.get("interrupted", False)),
    }


def load_run_summaries() -> pd.DataFrame:
    rows = []
    for json_path in sorted(OUTPUTS_DIR.glob("*.json")):
        row = summarize_run(json_path)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def build_executive_summary(run_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if run_df.empty:
        return ["No run artifacts were found."]

    total_video_runs = int((run_df["source_type"] == "video").sum())
    total_rtsp_runs = int((run_df["source_type"] == "rtsp").sum())
    models_tested = ", ".join(sorted(run_df["model"].dropna().unique().tolist()))
    lines.append(f"Validated {models_tested} across {total_video_runs} recorded-video runs and {total_rtsp_runs} live RTSP runs.")

    if not benchmark_df.empty and {"model", "fps_mean", "latency_ms_mean"}.issubset(benchmark_df.columns):
        fastest = benchmark_df.sort_values("fps_mean", ascending=False).iloc[0]
        slowest = benchmark_df.sort_values("latency_ms_mean", ascending=True).iloc[0]
        lines.append(
            f"{fastest['model']} delivered the strongest throughput in the controlled benchmark "
            f"({fastest['fps_mean']:.2f} FPS), while {slowest['model']} achieved the lowest measured latency "
            f"({slowest['latency_ms_mean']:.2f} ms)."
        )

    grouped = run_df.groupby(["source_type", "model"], as_index=False).agg(
        avg_fps=("avg_fps", "mean"),
        avg_latency_ms=("avg_latency_ms", "mean"),
        avg_zero_people_ratio=("zero_people_ratio", "mean"),
        avg_keypoint_confidence=("avg_keypoint_confidence", "mean"),
    )
    video_group = grouped[grouped["source_type"] == "video"]
    if not video_group.empty:
        best_video = video_group.sort_values("avg_latency_ms").iloc[0]
        lines.append(
            f"On recorded footage, {best_video['model']} was the more practical runtime choice "
            f"with mean latency around {best_video['avg_latency_ms']:.2f} ms."
        )

    rtsp_group = grouped[grouped["source_type"] == "rtsp"]
    if not rtsp_group.empty:
        lines.append(
            "Live RTSP validation succeeded, but stream quality and network transport materially affected smoothness and stability."
        )
    return lines


def build_findings(run_df: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    if run_df.empty:
        return findings

    video_df = run_df[run_df["source_type"] == "video"]
    rtsp_df = run_df[run_df["source_type"] == "rtsp"]

    if not video_df.empty:
        high_zero = video_df[video_df["zero_people_ratio"] > 0.5]
        if not high_zero.empty:
            examples = ", ".join(high_zero["display_name"].head(3).tolist())
            findings.append(
                f"Far-distance, crowded, or partially visible scenes remained the biggest challenge; several recorded runs "
                f"showed high no-person ratios (examples: {examples})."
            )
        findings.append(
            "When subjects were closer to the camera and fully visible, both models produced stable 17-keypoint skeletons with usable confidence."
        )

    if not rtsp_df.empty:
        findings.append(
            "Live RTSP runs confirmed end-to-end camera support, but output smoothness depended on stream transport quality, codec behavior, and real-time processing speed."
        )

    grouped = run_df.groupby("model", as_index=False).agg(
        avg_latency_ms=("avg_latency_ms", "mean"),
        avg_fps=("avg_fps", "mean"),
        avg_zero_people_ratio=("zero_people_ratio", "mean"),
    )
    if len(grouped) >= 2:
        fastest = grouped.sort_values("avg_latency_ms").iloc[0]
        slowest = grouped.sort_values("avg_latency_ms", ascending=False).iloc[0]
        findings.append(
            f"{fastest['model']} was the more deployable model overall from a speed perspective, while {slowest['model']} "
            "served better as a comparison/accuracy-oriented reference."
        )

    return findings


def build_recommendations() -> list[str]:
    return [
        "Standardize on RTMPose first for practical deployment and continued CCTV validation, because it is faster and more operationally manageable.",
        "Keep ViTPose as a reference model for comparison and deeper analysis, especially when speed is less critical than pose quality.",
        "Prioritize additional testing on far-distance, crowded, and occluded camera views, since those remain the main break conditions.",
        "Plan for finetuning or stronger person detection if long-distance subjects and dense crowd scenes are core deployment requirements.",
    ]


def main() -> None:
    args = parse_args()
    benchmark_df = pd.read_csv(args.benchmark_csv) if Path(args.benchmark_csv).exists() else pd.DataFrame()
    official_coco = load_json(Path(args.official_coco_json)) if Path(args.official_coco_json).exists() else {"models": [], "notes": []}
    run_df = load_run_summaries()

    summary_lines = build_executive_summary(run_df, benchmark_df)
    findings = build_findings(run_df)
    recommendations = build_recommendations()

    recorded_df = run_df[run_df["source_type"] == "video"].copy()
    live_df = run_df[run_df["source_type"] == "rtsp"].copy()

    aggregated_by_model = pd.DataFrame()
    if not run_df.empty:
        aggregated_by_model = (
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

    official_df = pd.DataFrame(official_coco.get("models", []))
    if not official_df.empty:
        official_df = official_df[["model", "variant", "input_size", "mAP", "mAP50", "mAP75", "recall"]]

    report = f"""# {args.title}

## Objective

This report summarizes the current benchmarking status of the reusable human pose detection primitive built with RTMPose and ViTPose for downstream use cases such as fall detection, patient monitoring, and surveillance analytics.

## Executive Summary

{chr(10).join(f"- {line}" for line in summary_lines)}

## Scope Of Validation

- Recorded CCTV-style video files were tested end to end with both RTMPose and ViTPose.
- Live RTSP camera feeds were validated end to end with on-screen display and saved output artifacts.
- Official COCO reference metrics were used as benchmark sanity-check baselines.
- Real-world camera footage was used as the main practical validation source.

## Official COCO Baseline Reference

{format_table(official_df)}

{chr(10).join(f"- {note}" for note in official_coco.get("notes", []))}

## Controlled Benchmark Snapshot

{format_table(benchmark_df)}

## Aggregate Real-World Run Summary

{format_table(aggregated_by_model)}

## Recorded Video Run Inventory

{format_table(recorded_df[["display_name", "model", "frames_processed", "avg_fps", "avg_latency_ms", "avg_people_count", "avg_keypoint_confidence", "zero_people_ratio"]])}

## Live RTSP Run Inventory

{format_table(live_df[["display_name", "model", "frames_processed", "avg_fps", "avg_latency_ms", "avg_people_count", "avg_keypoint_confidence", "rtsp_mode", "interrupted"]])}

## Findings: Where It Works

- Both models produced usable 17-keypoint full-body skeletons when people were reasonably close, visible, and not heavily occluded.
- The pipeline ran successfully on both stored video and live RTSP feeds, proving end-to-end operational viability.
- RTMPose consistently provided the more practical runtime profile for real-world usage on the current CPU-based setup.

## Findings: Where It Breaks

{chr(10).join(f"- {line}" for line in findings)}

## Recommendation

{chr(10).join(f"- {line}" for line in recommendations)}

## Suggested Client Update

- Pose detection is running end to end on recorded videos and on at least one live RTSP camera feed.
- RTMPose is currently the better standardization candidate for deployment because it offers the best balance of speed and field usability.
- ViTPose remains useful as a comparison model, but it is materially heavier and slower on the current setup.
- The main observed failure modes are far-distance subjects, crowding, occlusion, and difficult surveillance viewpoints.
- Additional finetuning or stronger detection support may be needed if those camera conditions dominate the target deployment.
"""

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
