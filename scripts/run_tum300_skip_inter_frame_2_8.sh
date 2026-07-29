#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
GPU="${GPU:-5}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/tum300_skip_inter_frame_2_8_2plus8}"

run_case() {
  local case_name="$1"
  local skip_blocks="$2"
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
    --dataset tum_dynamic \
    --output-dir "$temp_output" \
    --max-frames-per-seq 300 \
    --window-size 0 \
    --checkpoint "$ROOT/checkpoints/vggt_omega_1b_512.pt" \
    --skip-inter-frame-attention-blocks "$skip_blocks" \
    --overwrite \
    --eval \
    > "$log" 2>&1

  for artifact in \
    "$temp_output/tum_dynamic/_summary_complete_scale_shift.json" \
    "$temp_output/tum_dynamic/_summary_scale_shift.json" \
    "$temp_output/tum_dynamic/_sequence_metrics_scale_shift.csv" \
    "$temp_output/tum_dynamic/_summary_pose_auc.json"; do
    if [[ -f "$artifact" ]]; then
      cp "$artifact" "$summary_dir/$(basename "$artifact")"
    fi
  done
  rm -rf "$temp_output"
  "$PYTHON_BIN" "$ROOT/scripts/summarize_inter_frame_skip_runs.py" "$RUN_ROOT"
}

mkdir -p "$RUN_ROOT"
run_case skip_block_02 "2"
run_case skip_block_08 "8"
run_case skip_blocks_02_08 "2,8"
