#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_sintel_all_accel_5methods.sh <gpu: 5|6|7>}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/sintel_all_accel_5methods_20260723}"
SINTEL_ROOT="${SINTEL_ROOT:-$ROOT/../dataset/Sintel/training}"
PYTHON="${PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
SPARSE_REPO="${SPARSE_REPO:-$ROOT/../sparse-vggt}"
SPARSE_PYTHON="${SPARSE_PYTHON:-$SPARSE_REPO/.venv/bin/python}"

run_case() {
  local name="$1"
  local python_bin="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT:$SPARSE_REPO/src:$SPARSE_REPO/external/vggt:$SPARSE_REPO/external/SpargeAttn:$SPARSE_REPO" \
  PYTHONNOUSERSITE=1 \
  HF_HOME="$ROOT/.cache/huggingface" \
  TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" "$ROOT/inference/infer.py" \
    --device cuda \
    --dataset sintel \
    --dataset-root "$SINTEL_ROOT" \
    --window-size 0 \
    --frame-sample-mode first \
    --checkpoint "$ROOT/checkpoints/vggt_omega_1b_512.pt" \
    --output-dir "$RUN_ROOT/$name" \
    --eval \
    --eval-align scale_shift \
    --pose-eval-frames 0 \
    --pose-eval-seed 0 \
    "$@"
}

mkdir -p "$RUN_ROOT"
case "$GPU" in
  5)
    run_case sintel_all_baseline "$PYTHON" --omega-accelerator none
    ;;
  6)
    run_case sintel_all_fastvggt_spatial_r090 "$PYTHON" \
      --omega-accelerator none \
      --enable-token-merging \
      --token-merging-method spatial \
      --token-merging-ratio 0.9 \
      --token-merging-start 0
    run_case sintel_all_sparse_vggt "$SPARSE_PYTHON" \
      --omega-accelerator sparse_vggt \
      --sparse-vggt-sparse-ratio 0.5 \
      --sparse-vggt-pool-mode avg
    ;;
  7)
    run_case sintel_all_da_vggt "$PYTHON" \
      --omega-accelerator da_vggt \
      --da-vggt-max-frames 64 \
      --da-vggt-sampling-method fl_maxmin \
      --da-vggt-n-anchors 1 \
      --da-vggt-dino-batch-size 256 \
      --da-vggt-lambda-div 0.0
    run_case sintel_all_ours_layerwise_p0986_s0948 "$PYTHON" \
      --omega-accelerator none \
      --enable-token-merging \
      --token-merging-method frame_persistent_spatial \
      --token-merging-ratio 0.9 \
      --token-merging-layer-ratios 1-10:0.9,11-18:0.0,19-24:0.9 \
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
    echo "Expected GPU 5, 6, or 7; got $GPU" >&2
    exit 2
    ;;
esac
