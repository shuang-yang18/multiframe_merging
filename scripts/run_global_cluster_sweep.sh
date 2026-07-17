#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
DATASET="${DATASET:-tum_dynamic}"
MAX_FRAMES="${MAX_FRAMES:-300}"
RESTORE_LAYER="${RESTORE_LAYER:-24}"
TOKEN_MERGING_RATIO="${TOKEN_MERGING_RATIO:-0.9}"

cd "$ROOT"

run_one() {
  local threshold="$1"
  local max_size="$2"
  local tag="${threshold/./}"
  local output_name="${DATASET}_global_cluster_thr${tag}_max${max_size}_restore${RESTORE_LAYER}"

  echo "[$(date '+%F %T')] global_cluster dataset=${DATASET} threshold=${threshold} max_size=${max_size}"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" inference/infer.py \
    --dataset "$DATASET" \
    --output-dir "outputs/${output_name}" \
    --max-frames-per-seq "$MAX_FRAMES" \
    --window-size 0 \
    --checkpoint "${CHECKPOINT:-checkpoints/vggt_omega_1b_512.pt}" \
    --overwrite \
    --eval \
    --enable-token-merging \
    --token-merging-method frame_persistent_spatial \
    --token-merging-ratio "$TOKEN_MERGING_RATIO" \
    --token-merging-start 0 \
    --token-merging-frame-pool-stride "${POOL_STRIDE:-2}" \
    --token-merging-frame-segment-threshold "${SEGMENT_THRESHOLD:-0.9}" \
    --token-merging-frame-merge-threshold "${MERGE_THRESHOLD:-0.1}" \
    --token-merging-frame-alpha "${FRAME_ALPHA:-0.1}" \
    --token-merging-frame-max-window "${MAX_WINDOW:-20}" \
    --token-merging-frame-restore-layer "$RESTORE_LAYER" \
    --token-merging-frame-group-strategy global_cluster \
    --token-merging-frame-multi-max-group-size "$max_size" \
    --token-merging-frame-multi-pair-threshold "$threshold" \
    --token-merging-frame-multi-span-threshold "$threshold"

  "$PYTHON" scripts/export_frame_merge_groups.py \
    "outputs/${output_name}/${DATASET}" \
    "outputs/${output_name}/${DATASET}/_frame_merge_groups"
}

if [[ "$#" -gt 0 ]]; then
  if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 [threshold max_group_size]" >&2
    exit 2
  fi
  run_one "$1" "$2"
else
  run_one 0.98 4
  run_one 0.95 3
  run_one 0.98 3
fi
