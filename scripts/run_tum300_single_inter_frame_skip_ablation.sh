#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
GPU="${GPU:-5}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/tum300_firstseq_single_inter_frame_skip_ablation}"
SEQUENCE="rgbd_dataset_freiburg3_sitting_halfsphere"

mkdir -p "$RUN_ROOT"

run_case() {
  local case_name="$1"
  local skip_block="$2"
  local case_root="$RUN_ROOT/$case_name"
  local temp_output="$case_root/_temporary"
  local summary_dir="$case_root/summaries"
  local log="$case_root/run.log"
  local skip_args=()

  mkdir -p "$summary_dir"
  rm -rf "$temp_output"
  if [[ -n "$skip_block" ]]; then
    skip_args=(--skip-inter-frame-attention-blocks "$skip_block")
  fi

  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HOME="$ROOT/.cache/huggingface" \
  TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub" \
  "$PYTHON_BIN" "$ROOT/inference/infer.py" \
    --dataset tum_dynamic \
    --sequences "$SEQUENCE" \
    --output-dir "$temp_output" \
    --max-frames-per-seq 300 \
    --window-size 0 \
    --checkpoint "$ROOT/checkpoints/vggt_omega_1b_512.pt" \
    --overwrite \
    --eval \
    "${skip_args[@]}" \
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
  "$PYTHON_BIN" "$ROOT/scripts/summarize_single_layer_skip.py" "$RUN_ROOT"
}

run_case baseline ""
for block_idx in $(seq 0 23); do
  run_case "skip_block_$(printf '%02d' "$block_idx")" "$block_idx"
done
