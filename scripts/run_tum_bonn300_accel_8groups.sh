#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_tum_bonn300_accel_8groups.sh <gpu: 5|6|7>}"
RUN_ROOT="${RUN_ROOT:-outputs/tum_bonn300_accel_8groups_20260722}"
CHECKPOINT="${CHECKPOINT:-checkpoints/vggt_omega_1b_512.pt}"
FAST_PYTHON="${FAST_PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
SPARSE_PYTHON="${SPARSE_PYTHON:-/data/mmc_syang/sparse-vggt/.venv/bin/python}"
SPARSE_REPO="${SPARSE_REPO:-/data/mmc_syang/sparse-vggt}"

BONN_ROOT="${BONN_ROOT:-datasets/Bonn/rgbd_bonn_dataset}"
TUM_ROOT="${TUM_ROOT:-datasets/TUM-Dynamics}"
MAX_FRAMES="${MAX_FRAMES:-300}"
POSE_EVAL_FRAMES="${POSE_EVAL_FRAMES:-10}"
POSE_EVAL_SEED="${POSE_EVAL_SEED:-0}"
OVERWRITE="${OVERWRITE:-1}"

FASTVGGT_RATIO="${FASTVGGT_RATIO:-0.9}"
DA_VGGT_MAX_FRAMES="${DA_VGGT_MAX_FRAMES:-64}"
DA_VGGT_SAMPLING_METHOD="${DA_VGGT_SAMPLING_METHOD:-fl_maxmin}"
DA_VGGT_N_ANCHORS="${DA_VGGT_N_ANCHORS:-1}"
DA_VGGT_LAMBDA_DIV="${DA_VGGT_LAMBDA_DIV:-0.0}"
SPARSE_VGGT_RATIO="${SPARSE_VGGT_RATIO:-0.5}"
SPARSE_VGGT_POOL_MODE="${SPARSE_VGGT_POOL_MODE:-avg}"

cd "$ROOT"
mkdir -p outputs/logs "$RUN_ROOT"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

COMMON_ARGS=(
  --max-frames-per-seq "$MAX_FRAMES"
  --window-size 0
  --checkpoint "$CHECKPOINT"
  --eval
  --pose-eval-frames "$POSE_EVAL_FRAMES"
  --pose-eval-seed "$POSE_EVAL_SEED"
)
if [[ "$OVERWRITE" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

run_case() {
  local dataset="$1"
  local dataset_root="$2"
  local method="$3"
  local python_bin="$4"
  local output_name="$5"
  shift 5

  echo "[$(date '+%F %T')] start dataset=${dataset} method=${method} gpu=${GPU} output=${RUN_ROOT}/${output_name}"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT:$SPARSE_REPO/src:$SPARSE_REPO/external/vggt:$SPARSE_REPO/external/SpargeAttn:$SPARSE_REPO" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" inference/infer.py \
    --dataset "$dataset" \
    --dataset-root "$dataset_root" \
    --output-dir "${RUN_ROOT}/${output_name}" \
    "${COMMON_ARGS[@]}" \
    "$@"
  echo "[$(date '+%F %T')] done dataset=${dataset} method=${method} gpu=${GPU} output=${RUN_ROOT}/${output_name}"
}

run_bonn_baseline() {
  run_case bonn "$BONN_ROOT" baseline "$FAST_PYTHON" bonn300_baseline \
    --bonn-rgb-dir rgb \
    --bonn-depth-dir depth \
    --omega-accelerator none
}

run_bonn_fastvggt() {
  run_case bonn "$BONN_ROOT" fastvggt "$FAST_PYTHON" bonn300_fastvggt_r090 \
    --bonn-rgb-dir rgb \
    --bonn-depth-dir depth \
    --omega-accelerator none \
    --enable-token-merging \
    --token-merging-method spatial \
    --token-merging-ratio "$FASTVGGT_RATIO" \
    --token-merging-start 0
}

run_bonn_sparse() {
  run_case bonn "$BONN_ROOT" sparse_vggt "$SPARSE_PYTHON" bonn300_sparse_vggt \
    --bonn-rgb-dir rgb \
    --bonn-depth-dir depth \
    --omega-accelerator sparse_vggt \
    --sparse-vggt-sparse-ratio "$SPARSE_VGGT_RATIO" \
    --sparse-vggt-pool-mode "$SPARSE_VGGT_POOL_MODE"
}

run_bonn_da() {
  run_case bonn "$BONN_ROOT" da_vggt "$FAST_PYTHON" bonn300_da_vggt \
    --bonn-rgb-dir rgb \
    --bonn-depth-dir depth \
    --omega-accelerator da_vggt \
    --da-vggt-max-frames "$DA_VGGT_MAX_FRAMES" \
    --da-vggt-sampling-method "$DA_VGGT_SAMPLING_METHOD" \
    --da-vggt-n-anchors "$DA_VGGT_N_ANCHORS" \
    --da-vggt-lambda-div "$DA_VGGT_LAMBDA_DIV"
}

run_tum_baseline() {
  run_case tum_dynamic "$TUM_ROOT" baseline "$FAST_PYTHON" tum300_baseline \
    --omega-accelerator none
}

run_tum_fastvggt() {
  run_case tum_dynamic "$TUM_ROOT" fastvggt "$FAST_PYTHON" tum300_fastvggt_r090 \
    --omega-accelerator none \
    --enable-token-merging \
    --token-merging-method spatial \
    --token-merging-ratio "$FASTVGGT_RATIO" \
    --token-merging-start 0
}

run_tum_sparse() {
  run_case tum_dynamic "$TUM_ROOT" sparse_vggt "$SPARSE_PYTHON" tum300_sparse_vggt \
    --omega-accelerator sparse_vggt \
    --sparse-vggt-sparse-ratio "$SPARSE_VGGT_RATIO" \
    --sparse-vggt-pool-mode "$SPARSE_VGGT_POOL_MODE"
}

run_tum_da() {
  run_case tum_dynamic "$TUM_ROOT" da_vggt "$FAST_PYTHON" tum300_da_vggt \
    --omega-accelerator da_vggt \
    --da-vggt-max-frames "$DA_VGGT_MAX_FRAMES" \
    --da-vggt-sampling-method "$DA_VGGT_SAMPLING_METHOD" \
    --da-vggt-n-anchors "$DA_VGGT_N_ANCHORS" \
    --da-vggt-lambda-div "$DA_VGGT_LAMBDA_DIV"
}

case "$GPU" in
  5)
    run_bonn_baseline
    run_bonn_fastvggt
    ;;
  6)
    run_bonn_sparse
    run_bonn_da
    ;;
  7)
    run_tum_baseline
    run_tum_fastvggt
    run_tum_sparse
    run_tum_da
    ;;
  *)
    echo "Expected GPU 5, 6, or 7; got ${GPU}" >&2
    exit 2
    ;;
esac
