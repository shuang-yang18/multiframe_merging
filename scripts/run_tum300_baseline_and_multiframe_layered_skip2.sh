#!/usr/bin/env bash
set -euo pipefail

# Run the original VGGT baseline and the migrated Omega configuration on the
# same TUM300 protocol. Usage: bash scripts/run_tum300_baseline_and_multiframe_layered_skip2.sh GPU OUTPUT_ROOT
GPU=${1:?GPU index required}
OUTPUT_ROOT=${2:?output root required}
CHECKPOINT=${VGGT_CHECKPOINT:-/data/mmc_syang/FastVGGT/ckpt/model_tracker_fixed_e20.pt}
DATASET_ROOT=${TUM_ROOT:-/data/mmc_syang/dataset/TUM-Dynamics}
PYTHON_BIN=${VGGT_PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}

COMMON=(
  --dataset tum_dynamic
  --dataset-root "$DATASET_ROOT"
  --checkpoint "$CHECKPOINT"
  --device cuda
  --window-size 0
  --max-frames-per-seq 300
  --frame-sample-mode first
  --inter-frame-attention global
  --eval-align scale_shift
  --overwrite
)

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference/infer_vggt.py \
  "${COMMON[@]}" \
  --output-dir "$OUTPUT_ROOT/tum300_vggt_baseline"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" inference/infer_vggt.py \
  "${COMMON[@]}" \
  --output-dir "$OUTPUT_ROOT/tum300_vggt_multiframe_layered_skip2" \
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
  --skip-global-attention-blocks 2
