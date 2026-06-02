#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV_PYTHON="${SCRIPT_DIR}/.pose-cpu/bin/python"
if [[ -x "${DEFAULT_VENV_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_VENV_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_rtsp.sh <rtmpose|vitpose> <rtsp_url> [camera_name] [cpu|auto|cuda:0]"
  echo "Example: bash run_rtsp.sh rtmpose 'rtsp://user:pass@ip:554/path' vacron_gate cpu"
  exit 1
fi

POSE_MODEL="$1"
RTSP_URL="$2"
CAMERA_NAME="${3:-vacron_live}"
DEVICE="${4:-cpu}"
shift $(( $# >= 4 ? 4 : $# ))
EXTRA_ARGS=("$@")

case "${POSE_MODEL}" in
  rtmpose|vitpose)
    ;;
  *)
    echo "Unsupported pose model: ${POSE_MODEL}"
    echo "Supported models: rtmpose, vitpose"
    exit 1
    ;;
esac

if [[ "${POSE_MODEL}" == "vitpose" ]]; then
  if ! "${PYTHON_BIN}" -c "import mmpretrain" >/dev/null 2>&1; then
    echo "ViTPose requires mmpretrain, but it is not installed in the selected Python environment."
    echo "Run: bash setup_vitpose.sh"
    exit 1
  fi
fi

echo "Running live RTSP pose detection"
echo "  Camera  : ${CAMERA_NAME}"
echo "  Model   : ${POSE_MODEL}"
echo "  Device  : ${DEVICE}"
echo "  Python  : ${PYTHON_BIN}"
echo "  Display : enabled"
echo "  RTSP    : smooth capture mode"
echo "  Outputs : timestamped MP4 + JSON in outputs/"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/rtsp_demo.py" \
  --rtsp-url "${RTSP_URL}" \
  --pose-model "${POSE_MODEL}" \
  --device "${DEVICE}" \
  --camera-name "${CAMERA_NAME}" \
  --rtsp-mode smooth \
  "${EXTRA_ARGS[@]}"
