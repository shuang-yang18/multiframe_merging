#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_tum300_dynamic_spatial_fastvggt.sh <gpu> <all|middle|late|middle_late>}"
SCHEDULE="${2:?usage: run_tum300_dynamic_spatial_fastvggt.sh <gpu> <all|middle|late|middle_late>}"
TUM_ROOT="${TUM_ROOT:-$ROOT/../dataset/TUM-Dynamics}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/tum300_dynamic_spatial_fastvggt_20260724}"
MERGE_RATIO="${MERGE_RATIO:-0.9}"

case "$SCHEDULE" in
  all|middle|late|middle_late) ;;
  *)
    echo "Unknown dynamic FastVGGT schedule: $SCHEDULE" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
HF_HOME="$ROOT/.cache/huggingface" \
TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" inference/infer.py \
  --device cuda \
  --dataset tum_dynamic \
  --dataset-root "$TUM_ROOT" \
  --max-frames-per-seq 300 \
  --window-size 0 \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir "$RUN_ROOT/tum300_dynamic_spatial_r${MERGE_RATIO}_${SCHEDULE}" \
  --overwrite \
  --eval \
  --eval-align scale_shift \
  --pose-eval-frames 0 \
  --pose-eval-seed 0 \
  --enable-token-merging \
  --token-merging-method dynamic_spatial \
  --token-merging-start 5 \
  --token-merging-ratio "$MERGE_RATIO" \
  --dynamic-fastvggt-schedule "$SCHEDULE"
