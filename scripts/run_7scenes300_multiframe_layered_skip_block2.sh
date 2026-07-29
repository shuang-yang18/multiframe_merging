#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
GPU="${GPU:-5}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/7scenes_test300_multiframe_p0986_s0948_layered_r090_skip_block2}"
TEMP_OUTPUT="$RUN_ROOT/_temporary"
SUMMARY_DIR="$RUN_ROOT/summaries"
LOG="$RUN_ROOT/run.log"
SCHEDULE="1-10:0.9,11-18:0.0,19-24:0.9"

mkdir -p "$RUN_ROOT" "$SUMMARY_DIR"
rm -rf "$TEMP_OUTPUT"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME="$ROOT/.cache/huggingface" \
TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub" \
"$PYTHON_BIN" "$ROOT/inference/infer.py" \
  --dataset 7scenes \
  --seven-scenes-split test \
  --output-dir "$TEMP_OUTPUT" \
  --max-frames-per-seq 300 \
  --window-size 0 \
  --checkpoint "$ROOT/checkpoints/vggt_omega_1b_512.pt" \
  --overwrite \
  --eval \
  --skip-inter-frame-attention-blocks 2 \
  --enable-token-merging \
  --token-merging-method frame_persistent_spatial \
  --token-merging-ratio 0.9 \
  --token-merging-layer-ratios "$SCHEDULE" \
  --token-merging-start 0 \
  --token-merging-frame-pool-stride 2 \
  --token-merging-frame-segment-threshold 0.9 \
  --token-merging-frame-merge-threshold 0.1 \
  --token-merging-frame-alpha 0.1 \
  --token-merging-frame-max-window 20 \
  --token-merging-frame-restore-layer 24 \
  --token-merging-frame-multi-max-group-size 4 \
  --token-merging-frame-multi-pair-threshold 0.986 \
  --token-merging-frame-multi-span-threshold 0.948 \
  > "$LOG" 2>&1

for artifact in \
  "$TEMP_OUTPUT/7scenes/_summary_complete_scale_shift.json" \
  "$TEMP_OUTPUT/7scenes/_summary_scale_shift.json" \
  "$TEMP_OUTPUT/7scenes/_sequence_metrics_scale_shift.csv" \
  "$TEMP_OUTPUT/7scenes/_summary_pose_auc.json"; do
  if [[ -f "$artifact" ]]; then
    cp "$artifact" "$SUMMARY_DIR/$(basename "$artifact")"
  fi
done
rm -rf "$TEMP_OUTPUT"
