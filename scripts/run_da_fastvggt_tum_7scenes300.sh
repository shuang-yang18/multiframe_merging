#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
GPU="${GPU:-4}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/da_fastvggt_r090_tum300_7scenes_test300}"

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
    --omega-accelerator da_vggt \
    --da-vggt-max-frames 64 \
    --da-vggt-sampling-method fl_maxmin \
    --da-vggt-n-anchors 1 \
    --da-vggt-lambda-div 0.0 \
    --enable-token-merging \
    --token-merging-method spatial \
    --token-merging-ratio 0.9 \
    --token-merging-start 0 \
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
