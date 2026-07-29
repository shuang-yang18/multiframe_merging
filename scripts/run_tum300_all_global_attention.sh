#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
GPU="${GPU:-5}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/tum300_baseline_all_global_attention}"
TEMP_OUTPUT="$RUN_ROOT/_temporary"
SUMMARY_DIR="$RUN_ROOT/summaries"
LOG="$RUN_ROOT/run.log"

mkdir -p "$RUN_ROOT" "$SUMMARY_DIR"
rm -rf "$TEMP_OUTPUT"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME="$ROOT/.cache/huggingface" \
TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub" \
"$PYTHON_BIN" "$ROOT/inference/infer.py" \
  --dataset tum_dynamic \
  --output-dir "$TEMP_OUTPUT" \
  --max-frames-per-seq 300 \
  --window-size 0 \
  --checkpoint "$ROOT/checkpoints/vggt_omega_1b_512.pt" \
  --inter-frame-attention global \
  --overwrite \
  --eval \
  > "$LOG" 2>&1

for artifact in \
  "$TEMP_OUTPUT/tum_dynamic/_summary_complete_scale_shift.json" \
  "$TEMP_OUTPUT/tum_dynamic/_summary_scale_shift.json" \
  "$TEMP_OUTPUT/tum_dynamic/_sequence_metrics_scale_shift.csv" \
  "$TEMP_OUTPUT/tum_dynamic/_summary_pose_auc.json"; do
  if [[ -f "$artifact" ]]; then
    cp "$artifact" "$SUMMARY_DIR/$(basename "$artifact")"
  fi
done
rm -rf "$TEMP_OUTPUT"
