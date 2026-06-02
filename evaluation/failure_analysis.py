from __future__ import annotations

from pathlib import Path

import pandas as pd


def analyze_failure_tags(metrics_csv: str | Path, scene_tags_csv: str | Path, output_csv: str | Path) -> pd.DataFrame:
    metrics_df = pd.read_csv(metrics_csv)
    tags_df = pd.read_csv(scene_tags_csv)

    merged = metrics_df.merge(tags_df, on="image_id", how="inner")
    summary = (
        merged.groupby("tag")
        .agg(
            images=("image_id", "nunique"),
            mean_people=("num_people", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            mean_pose_score=("mean_pose_score", "mean"),
        )
        .reset_index()
        .sort_values("mean_pose_score")
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    return summary
