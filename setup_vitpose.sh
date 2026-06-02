#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV_PYTHON="${SCRIPT_DIR}/.pose-cpu/bin/python"
if [[ -x "${DEFAULT_VENV_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_VENV_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

CHECKPOINTS_DIR="${SCRIPT_DIR}/checkpoints"
VITPOSE_CKPT="${CHECKPOINTS_DIR}/vitpose_base.pth"
TMP_CKPT="${CHECKPOINTS_DIR}/vitpose_base.pth.part"
VITPOSE_URL="https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192-0b8234ea_20230407.pth"

mkdir -p "${CHECKPOINTS_DIR}"

echo "Preparing ViTPose support"
echo "  Python      : ${PYTHON_BIN}"
echo "  Checkpoint  : ${VITPOSE_CKPT}"

"${PYTHON_BIN}" -m pip install "mmpretrain>=1.0.0"

if [[ ! -f "${VITPOSE_CKPT}" ]]; then
  "${PYTHON_BIN}" - <<PY
import urllib.request
import os
url = "${VITPOSE_URL}"
tmp_dst = "${TMP_CKPT}"
dst = "${VITPOSE_CKPT}"
print(f"Downloading {url}")
urllib.request.urlretrieve(url, tmp_dst)
if os.path.getsize(tmp_dst) <= 0:
    raise RuntimeError("Downloaded checkpoint is empty")
os.replace(tmp_dst, dst)
print(f"Saved {dst}")
PY
else
  echo "Checkpoint already exists: ${VITPOSE_CKPT}"
fi

echo "ViTPose setup complete."
echo "Run: bash run_video.sh pose1.mp4 vitpose"
