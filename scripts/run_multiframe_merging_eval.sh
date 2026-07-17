#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
DATASET="${1:?dataset required: tum_dynamic|7scenes}"
OUTPUT_NAME="${2:?output name required}"

PAIR_THRESHOLD="${PAIR_THRESHOLD:-0.98}"
SPAN_THRESHOLD="${SPAN_THRESHOLD:-0.95}"
RESTORE_LAYER="${RESTORE_LAYER:-24}"
TOKEN_MERGING_RATIO="${TOKEN_MERGING_RATIO:-0.9}"
MAX_GROUP_SIZE="${MAX_GROUP_SIZE:-4}"
MAX_FRAMES="${MAX_FRAMES:-300}"
POOL_STRIDE="${POOL_STRIDE:-2}"

EXTRA_ARGS=()
if [[ "$DATASET" == "7scenes" ]]; then
  EXTRA_ARGS=(
    --dataset-root "${SEVEN_SCENES_ROOT:-datasets/7scenes/test}"
    --seven-scenes-split test
  )
fi

cd "$ROOT"
echo "[$(date '+%F %T')] multiframe_merging dataset=${DATASET} output=${OUTPUT_NAME} gpu=${GPU}"
echo "  ratio=${TOKEN_MERGING_RATIO} max_group=${MAX_GROUP_SIZE} pair=${PAIR_THRESHOLD} span=${SPAN_THRESHOLD} restore=${RESTORE_LAYER}"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" inference/infer.py \
  --dataset "$DATASET" \
  --output-dir "outputs/${OUTPUT_NAME}" \
  --max-frames-per-seq "$MAX_FRAMES" \
  --window-size 0 \
  --checkpoint "${CHECKPOINT:-checkpoints/vggt_omega_1b_512.pt}" \
  --overwrite \
  --eval \
  --enable-token-merging \
  --token-merging-method frame_persistent_spatial \
  --token-merging-ratio "$TOKEN_MERGING_RATIO" \
  --token-merging-start 0 \
  --token-merging-frame-pool-stride "$POOL_STRIDE" \
  --token-merging-frame-segment-threshold "${SEGMENT_THRESHOLD:-0.9}" \
  --token-merging-frame-merge-threshold "${MERGE_THRESHOLD:-0.1}" \
  --token-merging-frame-alpha "${FRAME_ALPHA:-0.1}" \
  --token-merging-frame-max-window "${MAX_WINDOW:-20}" \
  --token-merging-frame-restore-layer "$RESTORE_LAYER" \
  --token-merging-frame-multi-max-group-size "$MAX_GROUP_SIZE" \
  --token-merging-frame-multi-pair-threshold "$PAIR_THRESHOLD" \
  --token-merging-frame-multi-span-threshold "$SPAN_THRESHOLD" \
  "${EXTRA_ARGS[@]}"
