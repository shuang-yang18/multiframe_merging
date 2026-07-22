#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-5}"
RUN_ROOT="${RUN_ROOT:-outputs/vggt_10frame_baseline}"
MAX_FRAMES="${MAX_FRAMES:-10}"
CHECKPOINT="${CHECKPOINT:-checkpoints/vggt_omega_1b_512.pt}"
TUM_ROOT="${TUM_ROOT:-datasets/TUM-Dynamics}"
SEVEN_SCENES_ROOT="${SEVEN_SCENES_ROOT:-datasets/7scenes/test}"

cd "$ROOT"
mkdir -p "$RUN_ROOT"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

run_case() {
  local dataset="$1"
  local dataset_root="$2"
  local output_name="$3"
  shift 3

  echo "[$(date '+%F %T')] dataset=${dataset} output=${output_name} gpu=${GPU} max_frames=${MAX_FRAMES}"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" inference/infer.py \
    --dataset "$dataset" \
    --dataset-root "$dataset_root" \
    --output-dir "$RUN_ROOT/$output_name" \
    --max-frames-per-seq "$MAX_FRAMES" \
    --window-size 0 \
    --checkpoint "$CHECKPOINT" \
    --overwrite \
    --eval \
    "$@"
}

run_case tum_dynamic "$TUM_ROOT" tum10_omega
run_case 7scenes "$SEVEN_SCENES_ROOT" 7scenes_test10_omega --seven-scenes-split test

"$PYTHON" scripts/collect_vggt_10frame_metrics.py --run-root "$RUN_ROOT"
