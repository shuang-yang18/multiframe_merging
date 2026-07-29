#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
GPU="${GPU:-5}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/multiframe_p0986_s0948_layered_r090_frameonly_block2_tum300_7scenes_test300}"
SCHEDULE="1-10:0.9,11-18:0.0,19-24:0.9"

run_case() {
  local case_name="$1"
  local dataset="$2"
  shift 2
  local case_root="$RUN_ROOT/$case_name"
  local temp_output="$case_root/_temporary"
  local summary_dir="$case_root/summaries"
  local log="$case_root/run.log"

  mkdir -p "$summary_dir"
  rm -rf "$temp_output"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HOME="$ROOT/.cache/huggingface" \
  TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub" \
  "$PYTHON_BIN" "$ROOT/inference/infer.py" \
    --dataset "$dataset" \
    --output-dir "$temp_output" \
    --max-frames-per-seq 300 \
    --window-size 0 \
    --checkpoint "$ROOT/checkpoints/vggt_omega_1b_512.pt" \
    --overwrite \
    --eval \
    --frame-only-inter-frame-blocks 2 \
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
    "$@" \
    > "$log" 2>&1

  for artifact in \
    "$temp_output/$dataset/_summary_complete_scale_shift.json" \
    "$temp_output/$dataset/_summary_scale_shift.json" \
    "$temp_output/$dataset/_sequence_metrics_scale_shift.csv" \
    "$temp_output/$dataset/_summary_pose_auc.json"; do
    if [[ -f "$artifact" ]]; then
      cp "$artifact" "$summary_dir/$(basename "$artifact")"
    fi
  done
  rm -rf "$temp_output"
}

mkdir -p "$RUN_ROOT"
run_case tum300 tum_dynamic
run_case 7scenes_test300 7scenes --seven-scenes-split test
