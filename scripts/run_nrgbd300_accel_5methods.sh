#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_nrgbd300_accel_5methods.sh <gpu: 5|6|7>}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/nrgbd300_accel_5methods_20260722}"
NRGBD_ROOT="${NRGBD_ROOT:-/data/mmc_syang/dataset/NRGBD}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}"
FAST_PYTHON="${FAST_PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
SPARSE_REPO="${SPARSE_REPO:-$ROOT/../sparse-vggt}"
SPARSE_PYTHON="${SPARSE_PYTHON:-$SPARSE_REPO/.venv/bin/python}"

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

run_case() {
  local method="$1"
  local python_bin="$2"
  local output_name="$3"
  shift 3

  echo "[$(date '+%F %T')] start dataset=nrgbd method=${method} gpu=${GPU} output=${RUN_ROOT}/${output_name}"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT:$SPARSE_REPO/src:$SPARSE_REPO/external/vggt:$SPARSE_REPO/external/SpargeAttn:$SPARSE_REPO" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" inference/infer.py \
    --dataset nrgbd \
    --dataset-root "$NRGBD_ROOT" \
    --max-frames-per-seq 300 \
    --window-size 0 \
    --checkpoint "$CHECKPOINT" \
    --output-dir "${RUN_ROOT}/${output_name}" \
    --eval \
    --pose-eval-frames 0 \
    --pose-eval-seed 0 \
    --overwrite \
    "$@"
  echo "[$(date '+%F %T')] done dataset=nrgbd method=${method} gpu=${GPU} output=${RUN_ROOT}/${output_name}"
}

case "$GPU" in
  5)
    run_case baseline "$FAST_PYTHON" nrgbd300_baseline \
      --omega-accelerator none
    ;;
  6)
    run_case fastvggt "$FAST_PYTHON" nrgbd300_fastvggt_spatial_r090 \
      --omega-accelerator none \
      --enable-token-merging \
      --token-merging-method spatial \
      --token-merging-ratio 0.9 \
      --token-merging-start 0
    run_case sparse_vggt "$SPARSE_PYTHON" nrgbd300_sparse_vggt \
      --omega-accelerator sparse_vggt \
      --sparse-vggt-sparse-ratio 0.5 \
      --sparse-vggt-pool-mode avg
    ;;
  7)
    run_case da_vggt "$FAST_PYTHON" nrgbd300_da_vggt \
      --omega-accelerator da_vggt \
      --da-vggt-max-frames 64 \
      --da-vggt-sampling-method fl_maxmin \
      --da-vggt-n-anchors 1 \
      --da-vggt-lambda-div 0.0
    run_case ours "$FAST_PYTHON" nrgbd300_ours_layerwise_p0986_s0948 \
      --omega-accelerator none \
      --enable-token-merging \
      --token-merging-method frame_persistent_spatial \
      --token-merging-ratio 0.9 \
      --token-merging-layer-ratios '1-10:0.9,11-18:0.0,19-24:0.9' \
      --token-merging-start 0 \
      --token-merging-frame-restore-layer 24 \
      --token-merging-frame-alpha 0.1 \
      --token-merging-frame-segment-threshold 0.9 \
      --token-merging-frame-merge-threshold 0.1 \
      --token-merging-frame-max-window 20 \
      --token-merging-frame-pool-stride 2 \
      --token-merging-frame-multi-max-group-size 4 \
      --token-merging-frame-multi-pair-threshold 0.986 \
      --token-merging-frame-multi-span-threshold 0.948 \
      --token-merging-frame-group-strategy local
    ;;
  *)
    echo "Expected GPU 5, 6, or 7; got ${GPU}" >&2
    exit 2
    ;;
esac
