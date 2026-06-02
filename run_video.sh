#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEOS_DIR="${SCRIPT_DIR}/videos"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs"
DEFAULT_VENV_PYTHON="${SCRIPT_DIR}/.pose-cpu/bin/python"
if [[ -x "${DEFAULT_VENV_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_VENV_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_video.sh <video_filename> <rtmpose|vitpose> [cpu|auto|cuda:0]"
  echo "Example: bash run_video.sh pose1.mp4 rtmpose"
  echo "Example: bash run_video.sh pose1.mp4 vitpose cpu"
  exit 1
fi

VIDEO_NAME="$1"
POSE_MODEL="$2"
DEVICE="${3:-cpu}"
SOURCE_PATH="${VIDEOS_DIR}/${VIDEO_NAME}"

if [[ ! -f "${SOURCE_PATH}" ]]; then
  echo "Video not found: ${SOURCE_PATH}"
  echo "Put your input file inside: ${VIDEOS_DIR}"
  exit 1
fi

DET_CONFIG="${SCRIPT_DIR}/configs/mmpose_configs/rtmdet_nano_320-8xb32_coco-person.py"
DET_CKPT="${SCRIPT_DIR}/checkpoints/rtmdet_nano_person.pth"

case "${POSE_MODEL}" in
  rtmpose)
    POSE_CONFIG="${SCRIPT_DIR}/configs/mmpose_configs/rtmpose-m_8xb256-420e_coco-256x192.py"
    POSE_CKPT="${SCRIPT_DIR}/checkpoints/rtmpose_m.pth"
    ;;
  vitpose)
    POSE_CONFIG="${SCRIPT_DIR}/configs/mmpose_configs/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py"
    POSE_CKPT="${SCRIPT_DIR}/checkpoints/vitpose_base.pth"
    ;;
  *)
    echo "Unsupported pose model: ${POSE_MODEL}"
    echo "Supported models: rtmpose, vitpose"
    exit 1
    ;;
esac

RESIZE_MAX_SIDE="${RESIZE_MAX_SIDE:-0}"

if [[ ! -f "${DET_CONFIG}" ]]; then
  echo "Detector config not found: ${DET_CONFIG}"
  exit 1
fi

if [[ ! -f "${DET_CKPT}" ]]; then
  echo "Detector checkpoint not found: ${DET_CKPT}"
  exit 1
fi

if [[ ! -f "${POSE_CONFIG}" ]]; then
  echo "Pose config not found: ${POSE_CONFIG}"
  exit 1
fi

if [[ ! -f "${POSE_CKPT}" ]]; then
  echo "Pose checkpoint not found: ${POSE_CKPT}"
  echo "For ${POSE_MODEL}, place the checkpoint at: ${POSE_CKPT}"
  exit 1
fi

if [[ "${POSE_MODEL}" == "vitpose" ]]; then
  if ! "${PYTHON_BIN}" -c "import mmpretrain" >/dev/null 2>&1; then
    echo "ViTPose requires mmpretrain, but it is not installed in the selected Python environment."
    echo "Run: bash setup_vitpose.sh"
    echo "Or manually install: ${PYTHON_BIN} -m pip install mmpretrain>=1.0.0"
    exit 1
  fi
fi

mkdir -p "${OUTPUTS_DIR}"

VIDEO_STEM="${VIDEO_NAME%.*}"
OUTPUT_VIDEO="${OUTPUTS_DIR}/${VIDEO_STEM}_${POSE_MODEL}_output.mp4"
OUTPUT_JSON="${OUTPUTS_DIR}/${VIDEO_STEM}_${POSE_MODEL}_output.json"

echo "Running pose detection"
echo "  Input   : ${SOURCE_PATH}"
echo "  Model   : ${POSE_MODEL}"
echo "  Device  : ${DEVICE}"
echo "  Resize  : max-side ${RESIZE_MAX_SIDE}"
echo "  Python  : ${PYTHON_BIN}"
echo "  Video   : ${OUTPUT_VIDEO}"
echo "  JSON    : ${OUTPUT_JSON}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" \
  --source "${SOURCE_PATH}" \
  --source-type video \
  --device "${DEVICE}" \
  --resize-max-side "${RESIZE_MAX_SIDE}" \
  --run-label "${VIDEO_NAME}" \
  --pose-model "${POSE_MODEL}" \
  --det-config "${DET_CONFIG}" \
  --det-ckpt "${DET_CKPT}" \
  --pose-config "${POSE_CONFIG}" \
  --pose-ckpt "${POSE_CKPT}" \
  --output-video "${OUTPUT_VIDEO}" \
  --output-json "${OUTPUT_JSON}"
