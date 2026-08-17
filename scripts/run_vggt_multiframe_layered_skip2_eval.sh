#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_vggt_multiframe_layered_skip2_eval.sh GPU CHECKPOINT DATASET OUTPUT_DIR
# DATASET is tum_dynamic or 7scenes. The command performs inference and both
# Omega-compatible evaluations in one process.
GPU=${1:?GPU index required}
CHECKPOINT=${2:?VGGT checkpoint path required}
DATASET=${3:?dataset required: tum_dynamic or 7scenes}
OUTPUT_DIR=${4:?output directory required}

case "$DATASET" in
  tum_dynamic) DATASET_ROOT="${DATASET_ROOT:-../dataset/TUM-Dynamics}" ;;
  7scenes) DATASET_ROOT="${DATASET_ROOT:-../dataset/7scenes}" ;;
  *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

CUDA_VISIBLE_DEVICES="$GPU" python inference/infer_vggt.py \
  --dataset "$DATASET" \
  --dataset-root "$DATASET_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --window-size 0 \
  --max-frames-per-seq 300 \
  --frame-sample-mode first \
  --inter-frame-attention global \
  --enable-token-merging \
  --token-merging-method frame_persistent_spatial \
  --token-merging-ratio 0.9 \
  --token-merging-layer-ratios '1-10:0.9,11-18:0.0,19-24:0.9' \
  --token-merging-start 0 \
  --token-merging-frame-restore-layer 24 \
  --token-merging-frame-alpha 0.1 \
  --token-merging-frame-segment-threshold 0.9 \
  --token-merging-frame-merge-threshold 0.1 \
  --token-merging-frame-max-window 20 \
  --token-merging-frame-pool-stride 2 \
  --token-merging-frame-multi-max-group-size 4 \
  --token-merging-frame-multi-pair-threshold 0.986 \
  --token-merging-frame-multi-span-threshold 0.948 \
  --skip-global-attention-blocks 2 \
  --eval-align scale_shift
