from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Pose Detection Dashboard", layout="wide")
st.title("Human Pose Detection Dashboard")

benchmark_path = Path("benchmarks/benchmark_summary.csv")
eval_path = Path("outputs/coco_eval_summary.json")
run_json_path = Path("outputs/pose_output.json")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Benchmarks")
    if benchmark_path.exists():
        benchmark_df = pd.read_csv(benchmark_path)
        st.dataframe(benchmark_df, use_container_width=True)
        if "fps_mean" in benchmark_df.columns:
            st.bar_chart(benchmark_df.set_index("model")["fps_mean"])
    else:
        st.info("Run benchmark.py to populate benchmark results.")

with col2:
    st.subheader("Evaluation")
    if eval_path.exists():
        eval_series = pd.read_json(eval_path, typ="series")
        st.json(eval_series.to_dict())
    else:
        st.info("Run evaluate_coco.py to populate evaluation results.")

st.subheader("Sample Inference Output")
if run_json_path.exists():
    run_data = json.loads(run_json_path.read_text(encoding="utf-8"))
    st.json(run_data.get("run_summary", {}))
else:
    st.info("Run main.py to generate inference artifacts.")
