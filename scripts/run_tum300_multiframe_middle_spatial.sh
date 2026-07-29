#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
GPU="${GPU:-6}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/tum300_multiframe_k0_max4_p0986_s0948_middle_r090}"
TEMP_OUTPUT="$RUN_ROOT/_temporary"
SUMMARY_DIR="$RUN_ROOT/summaries"
LOG="$RUN_ROOT/run.log"

# First/last five 1-based aggregator layers use r=0; middle fourteen use r=0.9.
SCHEDULE="${SCHEDULE:-1-5:0.0,6-19:0.9,20-24:0.0}"

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
  --overwrite \
  --eval \
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

for summary in "$TEMP_OUTPUT/tum_dynamic/_summary_complete_scale_shift.json" "$TEMP_OUTPUT/tum_dynamic/_summary_scale_shift.json"; do
  if [[ -f "$summary" ]]; then
    cp "$summary" "$SUMMARY_DIR/$(basename "$summary")"
  fi
done
# Keep sequence-level depth rows and the pose summary before discarding heavy
# reconstruction artifacts from the temporary evaluation directory.
for metrics in "$TEMP_OUTPUT/tum_dynamic/_sequence_metrics_scale_shift.csv" "$TEMP_OUTPUT/tum_dynamic/_summary_pose_auc.json"; do
  if [[ -f "$metrics" ]]; then
    cp "$metrics" "$SUMMARY_DIR/$(basename "$metrics")"
  fi
done
rm -rf "$TEMP_OUTPUT"
