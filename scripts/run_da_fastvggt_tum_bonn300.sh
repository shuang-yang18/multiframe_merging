#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${GPU:-7}"
RUN_ROOT="${RUN_ROOT:-outputs/da_fastvggt_tum_bonn300_20260722}"
PYTHON="${PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/vggt_omega_1b_512.pt}"
TUM_ROOT="${TUM_ROOT:-datasets/TUM-Dynamics}"
BONN_ROOT="${BONN_ROOT:-datasets/Bonn/rgbd_bonn_dataset}"
MAX_FRAMES="${MAX_FRAMES:-300}"
POSE_EVAL_FRAMES="${POSE_EVAL_FRAMES:-10}"
POSE_EVAL_SEED="${POSE_EVAL_SEED:-0}"
FASTVGGT_RATIO="${FASTVGGT_RATIO:-0.9}"
DA_VGGT_MAX_FRAMES="${DA_VGGT_MAX_FRAMES:-64}"
DA_VGGT_SAMPLING_METHOD="${DA_VGGT_SAMPLING_METHOD:-fl_maxmin}"
DA_VGGT_N_ANCHORS="${DA_VGGT_N_ANCHORS:-1}"
DA_VGGT_LAMBDA_DIV="${DA_VGGT_LAMBDA_DIV:-0.0}"

cd "$ROOT"
mkdir -p "$RUN_ROOT" outputs/logs
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

COMMON_ARGS=(
  --max-frames-per-seq "$MAX_FRAMES"
  --window-size 0
  --checkpoint "$CHECKPOINT"
  --overwrite
  --eval
  --pose-eval-frames "$POSE_EVAL_FRAMES"
  --pose-eval-seed "$POSE_EVAL_SEED"
  --omega-accelerator da_vggt
  --da-vggt-max-frames "$DA_VGGT_MAX_FRAMES"
  --da-vggt-sampling-method "$DA_VGGT_SAMPLING_METHOD"
  --da-vggt-n-anchors "$DA_VGGT_N_ANCHORS"
  --da-vggt-lambda-div "$DA_VGGT_LAMBDA_DIV"
  --enable-token-merging
  --token-merging-method spatial
  --token-merging-ratio "$FASTVGGT_RATIO"
  --token-merging-start 0
)

run_case() {
  local dataset="$1"
  local dataset_root="$2"
  local output_name="$3"
  shift 3
  echo "[$(date '+%F %T')] start DA+FastVGGT dataset=${dataset} gpu=${GPU} output=${RUN_ROOT}/${output_name}"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" inference/infer.py \
    --dataset "$dataset" \
    --dataset-root "$dataset_root" \
    --output-dir "${RUN_ROOT}/${output_name}" \
    "${COMMON_ARGS[@]}" \
    "$@"
  echo "[$(date '+%F %T')] done DA+FastVGGT dataset=${dataset} gpu=${GPU} output=${RUN_ROOT}/${output_name}"
}

run_case tum_dynamic "$TUM_ROOT" tum300_da_fastvggt_r090
run_case bonn "$BONN_ROOT" bonn300_da_fastvggt_r090 --bonn-rgb-dir rgb --bonn-depth-dir depth
