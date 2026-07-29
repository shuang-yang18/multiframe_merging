#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_tum_baseline_auc_official.sh <gpu> <max_frames> <output_name>}"
MAX_FRAMES="${2:?usage: run_tum_baseline_auc_official.sh <gpu> <max_frames> <output_name>}"
OUTPUT_NAME="${3:?usage: run_tum_baseline_auc_official.sh <gpu> <max_frames> <output_name>}"

PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/vggt_omega_1b_512.pt}"
TUM_ROOT="${TUM_ROOT:-datasets/TUM-Dynamics}"
RUN_ROOT="${RUN_ROOT:-auc_eval_results/tum_official_auc}"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

echo "[$(date '+%F %T')] start TUM baseline max_frames=${MAX_FRAMES} gpu=${GPU} output=${RUN_ROOT}/${OUTPUT_NAME}"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" inference/infer.py \
  --dataset tum_dynamic \
  --dataset-root "$TUM_ROOT" \
  --output-dir "$RUN_ROOT/$OUTPUT_NAME" \
  --max-frames-per-seq "$MAX_FRAMES" \
  --window-size 0 \
  --checkpoint "$CHECKPOINT" \
  --overwrite \
  --eval \
  --pose-eval-frames 0

echo "[$(date '+%F %T')] done TUM baseline max_frames=${MAX_FRAMES} gpu=${GPU} output=${RUN_ROOT}/${OUTPUT_NAME}"
