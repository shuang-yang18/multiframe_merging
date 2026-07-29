#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: continue_bonn300_accel_8groups.sh <gpu: 5|6>}"
RUN_ROOT="${RUN_ROOT:-outputs/tum_bonn300_accel_8groups_20260722}"
CHECKPOINT="${CHECKPOINT:-checkpoints/vggt_omega_1b_512.pt}"
FAST_PYTHON="${FAST_PYTHON:-python}"
BONN_ROOT="${BONN_ROOT:-datasets/Bonn/rgbd_bonn_dataset}"
MAX_FRAMES="${MAX_FRAMES:-300}"
POSE_EVAL_FRAMES="${POSE_EVAL_FRAMES:-0}"
POSE_EVAL_SEED="${POSE_EVAL_SEED:-0}"
FASTVGGT_RATIO="${FASTVGGT_RATIO:-0.9}"
DA_VGGT_MAX_FRAMES="${DA_VGGT_MAX_FRAMES:-64}"
DA_VGGT_SAMPLING_METHOD="${DA_VGGT_SAMPLING_METHOD:-fl_maxmin}"
DA_VGGT_N_ANCHORS="${DA_VGGT_N_ANCHORS:-1}"
DA_VGGT_LAMBDA_DIV="${DA_VGGT_LAMBDA_DIV:-0.0}"

cd "$ROOT"
mkdir -p outputs/logs "$RUN_ROOT"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

eval_existing() {
  local output_name="$1"
  local summary_path="${RUN_ROOT}/${output_name}/bonn/bonn-complete-scale_shift.json"
  if [[ -f "$summary_path" || -f "${RUN_ROOT}/${output_name}/bonn/_summary_complete_scale_shift.json" ]]; then
    echo "[$(date '+%F %T')] skip existing Bonn eval output=${RUN_ROOT}/${output_name}"
    return
  fi
  echo "[$(date '+%F %T')] eval existing Bonn output=${RUN_ROOT}/${output_name}"
  "$FAST_PYTHON" inference/eval.py \
    --dataset bonn \
    --dataset-root "$BONN_ROOT" \
    --pred-dir "${RUN_ROOT}/${output_name}" \
    --output-dir "${RUN_ROOT}/${output_name}/bonn" \
    --bonn-rgb-dir rgb \
    --bonn-depth-dir depth \
    --max-frames-per-seq "$MAX_FRAMES" \
    --pose-eval-frames "$POSE_EVAL_FRAMES" \
    --pose-eval-seed "$POSE_EVAL_SEED"
}

run_infer_eval() {
  local method="$1"
  local output_name="$2"
  shift 2
  echo "[$(date '+%F %T')] start Bonn method=${method} gpu=${GPU} output=${RUN_ROOT}/${output_name}"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$FAST_PYTHON" inference/infer.py \
    --dataset bonn \
    --dataset-root "$BONN_ROOT" \
    --output-dir "${RUN_ROOT}/${output_name}" \
    --bonn-rgb-dir rgb \
    --bonn-depth-dir depth \
    --max-frames-per-seq "$MAX_FRAMES" \
    --window-size 0 \
    --checkpoint "$CHECKPOINT" \
    --overwrite \
    --eval \
    --pose-eval-frames "$POSE_EVAL_FRAMES" \
    --pose-eval-seed "$POSE_EVAL_SEED" \
    "$@"
  echo "[$(date '+%F %T')] done Bonn method=${method} gpu=${GPU} output=${RUN_ROOT}/${output_name}"
}

case "$GPU" in
  5)
    eval_existing bonn300_baseline
    run_infer_eval fastvggt bonn300_fastvggt_r090 \
      --omega-accelerator none \
      --enable-token-merging \
      --token-merging-method spatial \
      --token-merging-ratio "$FASTVGGT_RATIO" \
      --token-merging-start 0
    ;;
  6)
    eval_existing bonn300_sparse_vggt
    run_infer_eval da_vggt bonn300_da_vggt \
      --omega-accelerator da_vggt \
      --da-vggt-max-frames "$DA_VGGT_MAX_FRAMES" \
      --da-vggt-sampling-method "$DA_VGGT_SAMPLING_METHOD" \
      --da-vggt-n-anchors "$DA_VGGT_N_ANCHORS" \
      --da-vggt-lambda-div "$DA_VGGT_LAMBDA_DIV"
    ;;
  *)
    echo "Expected GPU 5 or 6; got ${GPU}" >&2
    exit 2
    ;;
esac
