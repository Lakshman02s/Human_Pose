# Human Pose Detection System

Production-style human pose detection stack built for downstream CV tasks such as fall detection, patient monitoring, human behavior analytics, CCTV intelligence, and surveillance pipelines.

## Features

- Multi-source inference: webcam, RTSP/IP cameras, CCTV footage, local video files
- 17-keypoint full-body human skeleton estimation for multiple people
- Pretrained pose backends: RTMPose (primary) and ViTPose (comparison)
- Person detection plus top-down pose inference pipeline
- CUDA-aware device selection with CPU fallback
- Visualization overlays: bounding boxes, skeletons, keypoints, IDs, FPS, latency
- Benchmarking: FPS, detector latency, pose latency, CPU, RAM, GPU memory
- COCO evaluation workflow with mAP and recall reporting
- Failure-analysis hooks for low light, occlusion, crowded scenes, overhead views, and long-range humans
- Extension points for tracking, smoothing, ONNX export, TensorRT deployment, and future fall detection logic

## Project Structure

```text
Human_Pose/
├── benchmarks/
├── configs/
├── datasets/
├── evaluation/
├── models/
├── notebooks/
├── outputs/
├── utils/
├── videos/
├── benchmark.py
├── evaluate_coco.py
├── generate_report.py
├── main.py
├── README.md
├── requirements.txt
├── rtsp_demo.py
├── streamlit_app.py
└── webcam_demo.py
```

## Architecture

Pipeline:

```text
Frame -> Human Detector -> Person BBoxes -> Pose Estimator -> Tracking/Smoothing -> Rendering -> JSON/Video Outputs
```

Core modules:

- `models/human_detector.py`: MMDetection person detector wrapper
- `models/pose_estimator.py`: MMPose top-down 17-keypoint inference wrapper
- `models/pipeline.py`: orchestration, tracking, smoothing, rendering integration
- `utils/visualization.py`: skeleton drawing and telemetry overlay
- `benchmark.py`: speed and resource benchmarking, including model comparison
- `evaluate_coco.py`: COCO keypoints evaluation
- `evaluation/failure_analysis.py`: scenario-level post-hoc analysis
- `generate_report.py`: benchmark and evaluation report generation
- `streamlit_app.py`: lightweight monitoring dashboard

## Setup

### 1. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install OpenMMLab dependencies with MIM

MMPose and MMDetection installations are tightly coupled to your local CUDA and PyTorch build. The safest production route is:

```bash
mim install "mmengine>=0.10.3"
mim install "mmcv>=2.1.0"
mim install "mmdet>=3.3.0"
mim install "mmpose>=1.3.1"
```

If you are deploying on a GPU machine, ensure your NVIDIA driver matches the installed CUDA runtime.

### 3. Download configs and checkpoints

You can either:

1. Clone `mmpose` and `mmdetection` repos locally and point to their config files.
2. Copy the exact config files into your local environment.
3. Update `configs/model_zoo.yaml` or pass explicit CLI paths.

Recommended assets:

- RTMDet person detector
- RTMPose-M or RTMPose-L for CCTV/edge tradeoff
- ViTPose-Base for accuracy comparison

### 4. Verify device support

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If `torch.cuda.is_available()` returns `False`, the pipeline automatically falls back to CPU.

## Inference

### Webcam

```bash
python3 main.py --source 0 --source-type webcam --display \
  --pose-model rtmpose \
  --det-config /path/to/mmdet_detector.py \
  --det-ckpt /path/to/detector.pth \
  --pose-config /path/to/rtmpose_config.py \
  --pose-ckpt /path/to/rtmpose_checkpoint.pth
```

### RTSP / IP camera

```bash
python3 main.py --source "rtsp://user:pass@camera-ip/stream" --source-type rtsp --display
```

Recommended live-camera runner:

```bash
bash run_rtsp.sh rtmpose "rtsp://user:pass@camera-ip/stream" vacron_live cpu
```

Or for ViTPose:

```bash
bash run_rtsp.sh vitpose "rtsp://user:pass@camera-ip/stream" vacron_live cpu
```

What this does:

- opens the RTSP stream live with on-screen display
- runs RTMDet + RTMPose or ViTPose on the live feed
- saves a timestamped output video in `outputs/`
- saves a timestamped JSON file in `outputs/`

### Local CCTV / video file

```bash
python3 main.py --source videos/cctv_clip.mp4 --source-type cctv \
  --output-video outputs/cctv_pose.mp4 \
  --output-json outputs/cctv_pose.json
```

### Simple video runner

Put all input videos inside `videos/`, then run:

```bash
bash run_video.sh pose1.mp4 rtmpose
```

or

```bash
bash run_video.sh pose1.mp4 vitpose
```

Optional third argument:

```bash
bash run_video.sh pose1.mp4 rtmpose cpu
```

What the script does for you:

- reads the input video from `videos/`
- automatically picks detector config/checkpoint
- automatically picks pose config/checkpoint for `rtmpose` or `vitpose`
- writes output video and JSON into `outputs/`

If you want to run `ViTPose`, prepare it once with:

```bash
bash setup_vitpose.sh
```

That installs `mmpretrain` and downloads the official ViTPose base checkpoint used by `run_video.sh`.

By default, `run_video.sh` keeps the original input resolution for processing and output. If you ever want to manually downscale only for faster experiments, you can override it like this:

```bash
RESIZE_MAX_SIDE=960 bash run_video.sh pose1.mp4 vitpose
```

Generated output names follow this pattern:

- `outputs/<video_stem>_<model>_output.mp4`
- `outputs/<video_stem>_<model>_output.json`

### Useful runtime options

```bash
--device auto
--det-score-thr 0.5
--kpt-score-thr 0.35
--fp16
--disable-tracking
--disable-smoothing
--max-frames 500
```

## Benchmarking

Compare RTMPose and ViTPose on the same source:

```bash
python3 benchmark.py --source videos/cctv_clip.mp4 --models rtmpose vitpose --max-frames 300
```

Artifacts:

- `benchmarks/benchmark_summary.json`
- `benchmarks/benchmark_summary.csv`

Captured metrics:

- Mean FPS
- Mean and P95 latency
- Detector latency
- Pose estimator latency
- Mean CPU usage
- Mean RAM usage
- Mean GPU memory usage

## COCO Evaluation

```bash
python3 evaluate_coco.py \
  --annotation-file datasets/coco/annotations/person_keypoints_val2017.json \
  --image-root datasets/coco/val2017 \
  --pose-model rtmpose \
  --output-json outputs/coco_predictions.json \
  --summary-json outputs/coco_eval_summary.json
```

Metrics reported:

- `mAP`
- `mAP50`
- `mAP75`
- `mAR`
- `precision` (reported as AP50 proxy)
- `recall` (reported as AR proxy)

## Failure Analysis Workflow

1. Run inference or evaluation and collect per-image or per-frame metrics.
2. Prepare a tag file such as `datasets/failure_tags/scene_tags.csv`.
3. Use `evaluation/failure_analysis.py` to aggregate failure-prone scenarios.

Recommended tags:

- `overhead_angle`
- `low_light`
- `occlusion`
- `crowded_scene`
- `partial_body`
- `long_distance`

## Report Generation

```bash
python3 generate_report.py \
  --benchmark-csv benchmarks/benchmark_summary.csv \
  --eval-json outputs/coco_eval_summary.json \
  --output-md outputs/final_report.md
```

## Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

## Benchmark Dashboard

A richer benchmark review UI is available through FastAPI + React.

Run it with:

```bash
source .pose-cpu/bin/activate
uvicorn dashboard_api:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The dashboard reads:

- `benchmarks/benchmark_summary.json`
- `benchmarks/history/*.json`
- `outputs/*_output.json`
- `outputs/*_output.mp4`

## JSON Output Contract

Each frame record contains:

- `frame_id`
- `timestamp_ms`
- `fps`
- `latency_ms`
- `detector_ms`
- `pose_ms`
- `cpu_percent`
- `ram_percent`
- `gpu_memory_mb`
- `poses[]`

Each pose contains:

- `track_id`
- `score`
- `bbox_xyxy`
- `keypoints_xyc`

## Optimization Path

This repository already includes the integration points for:

- Micro-batched frame processing via `PosePipeline.process_batch`
- FP16 gating through device capability checks
- ONNX export command generation in `utils/export.py`
- TensorRT deployment command generation in `utils/export.py`
- Lightweight multi-person tracking
- Temporal pose smoothing

Recommended production upgrades:

1. Add MMDeploy for formal ONNX/TensorRT graph export.
2. Use RTMPose-M for balanced edge deployment, RTMPose-L for server accuracy.
3. Replace the simple IoU tracker with ByteTrack or OC-SORT when long-lived IDs matter.
4. Add asynchronous decode/inference/render worker queues for high-throughput CCTV.
5. Add alerting/business-rule modules for fall detection or patient safety.

## Model Comparison Guidance

Expected practical tradeoff:

- `RTMPose`: better real-time throughput, strong edge suitability, easier CCTV deployment
- `ViTPose`: stronger accuracy potential, usually heavier and slower

For enterprise surveillance, start with RTMPose unless:

- fine-grained accuracy is more important than throughput
- server-side GPU budget is comfortable
- crowded scenes or long-distance people justify the extra capacity

## Suggested Report Template

Use benchmark and evaluation artifacts to generate a final decision report with:

1. Hardware and software environment
2. Model configurations and checkpoint versions
3. Quantitative benchmark table
4. COCO evaluation table
5. Failure-case screenshots
6. CCTV-specific qualitative observations
7. Recommendation for RTMPose vs ViTPose
8. Finetuning recommendation and data gaps

## Notes

- The OpenMMLab config paths in this repository are intentionally editable because local environments vary.
- The current code expects top-down 17-keypoint models compatible with COCO-style output.
- If your machine reports CUDA unavailable, check the NVIDIA driver and CUDA compatibility first.
