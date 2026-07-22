#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-outputs/bonn_tum300_accel_20260721}"
PYTHON="${PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
BONN_ROOT="${BONN_ROOT:-/data/mmc_syang/dataset/Bonn/rgbd_bonn_dataset}"
TUM_ROOT="${TUM_ROOT:-/data/mmc_syang/dataset/TUM-Dynamics}"

cd "$ROOT"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

eval_case() {
  local dataset="$1"
  local dataset_root="$2"
  local output_name="$3"
  shift 3

  echo "[$(date '+%F %T')] eval dataset=${dataset} output=${RUN_ROOT}/${output_name}"
  "$PYTHON" inference/eval.py \
    --dataset "$dataset" \
    --dataset-root "$dataset_root" \
    --pred-dir "${RUN_ROOT}/${output_name}" \
    --output-dir "${RUN_ROOT}/${output_name}/${dataset}" \
    --max-frames-per-seq 300 \
    "$@"
}

eval_case bonn "$BONN_ROOT" bonn300_omega --bonn-rgb-dir rgb --bonn-depth-dir depth
eval_case bonn "$BONN_ROOT" bonn300_fastvggt_r090 --bonn-rgb-dir rgb --bonn-depth-dir depth
eval_case bonn "$BONN_ROOT" bonn300_sparse_vggt --bonn-rgb-dir rgb --bonn-depth-dir depth
eval_case bonn "$BONN_ROOT" bonn300_da_vggt --bonn-rgb-dir rgb --bonn-depth-dir depth

eval_case tum_dynamic "$TUM_ROOT" tum300_omega
eval_case tum_dynamic "$TUM_ROOT" tum300_fastvggt_r090
eval_case tum_dynamic "$TUM_ROOT" tum300_sparse_vggt
eval_case tum_dynamic "$TUM_ROOT" tum300_da_vggt

"$PYTHON" scripts/collect_bonn_tum300_accel_metrics.py
