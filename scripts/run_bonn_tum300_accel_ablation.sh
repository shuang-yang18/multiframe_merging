#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_bonn_tum300_accel_ablation.sh <gpu>}"
RUN_ROOT="${RUN_ROOT:-outputs/bonn_tum300_accel_20260721}"
CHECKPOINT="${CHECKPOINT:-checkpoints/vggt_omega_1b_512.pt}"
FAST_PYTHON="${FAST_PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
SPARSE_PYTHON="${SPARSE_PYTHON:-/data/mmc_syang/sparse-vggt/.venv/bin/python}"

BONN_ROOT="${BONN_ROOT:-/data/mmc_syang/dataset/Bonn/rgbd_bonn_dataset}"
TUM_ROOT="${TUM_ROOT:-/data/mmc_syang/dataset/TUM-Dynamics}"
OVERWRITE="${OVERWRITE:-0}"
RUN_EVAL="${RUN_EVAL:-0}"

COMMON_ARGS=(
  --max-frames-per-seq 300
  --window-size 0
  --checkpoint "$CHECKPOINT"
)
if [[ "$OVERWRITE" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi
if [[ "$RUN_EVAL" == "1" ]]; then
  COMMON_ARGS+=(--eval)
fi

cd "$ROOT"
mkdir -p outputs/logs "$RUN_ROOT"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

run_case() {
  local dataset="$1"
  local dataset_root="$2"
  local method="$3"
  local python_bin="$4"
  local output_name="$5"
  shift 5

  echo "[$(date '+%F %T')] start dataset=${dataset} method=${method} gpu=${GPU} output=${RUN_ROOT}/${output_name}"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT:/data/mmc_syang/sparse-vggt" \
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

case "$GPU" in
  6)
    run_case bonn "$BONN_ROOT" omega "$FAST_PYTHON" bonn300_omega \
      --bonn-rgb-dir rgb \
      --bonn-depth-dir depth \
      --omega-accelerator none
    run_case bonn "$BONN_ROOT" fastvggt_r090 "$FAST_PYTHON" bonn300_fastvggt_r090 \
      --bonn-rgb-dir rgb \
      --bonn-depth-dir depth \
      --omega-accelerator none \
      --enable-token-merging \
      --token-merging-method spatial \
      --token-merging-ratio 0.9 \
      --token-merging-start 0
    run_case tum_dynamic "$TUM_ROOT" sparse_vggt "$SPARSE_PYTHON" tum300_sparse_vggt \
      --omega-accelerator sparse_vggt \
      --sparse-vggt-sparse-ratio 0.5 \
      --sparse-vggt-pool-mode avg
    run_case tum_dynamic "$TUM_ROOT" da_vggt "$FAST_PYTHON" tum300_da_vggt \
      --omega-accelerator da_vggt \
      --da-vggt-max-frames 64 \
      --da-vggt-sampling-method fl_maxmin \
      --da-vggt-n-anchors 1
    ;;
  7)
    run_case tum_dynamic "$TUM_ROOT" omega "$FAST_PYTHON" tum300_omega \
      --omega-accelerator none
    run_case tum_dynamic "$TUM_ROOT" fastvggt_r090 "$FAST_PYTHON" tum300_fastvggt_r090 \
      --omega-accelerator none \
      --enable-token-merging \
      --token-merging-method spatial \
      --token-merging-ratio 0.9 \
      --token-merging-start 0
    run_case bonn "$BONN_ROOT" sparse_vggt "$SPARSE_PYTHON" bonn300_sparse_vggt \
      --bonn-rgb-dir rgb \
      --bonn-depth-dir depth \
      --omega-accelerator sparse_vggt \
      --sparse-vggt-sparse-ratio 0.5 \
      --sparse-vggt-pool-mode avg
    run_case bonn "$BONN_ROOT" da_vggt "$FAST_PYTHON" bonn300_da_vggt \
      --bonn-rgb-dir rgb \
      --bonn-depth-dir depth \
      --omega-accelerator da_vggt \
      --da-vggt-max-frames 64 \
      --da-vggt-sampling-method fl_maxmin \
      --da-vggt-n-anchors 1
    ;;
  *)
    echo "This ablation runner is intentionally scoped to GPU 6 or 7; got ${GPU}" >&2
    exit 2
    ;;
esac
