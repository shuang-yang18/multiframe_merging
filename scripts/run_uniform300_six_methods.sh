#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_uniform300_six_methods.sh <gpu: 2|3|4|5|6|7>}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/01}"
FAST_PYTHON="${FAST_PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
SPARSE_REPO="${SPARSE_REPO:-/data/mmc_syang/sparse-vggt}"
SPARSE_PYTHON="${SPARSE_PYTHON:-$SPARSE_REPO/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}"

TUM_ROOT="${TUM_ROOT:-/data/mmc_syang/dataset/TUM-Dynamics}"
SEVEN_SCENES_ROOT="${SEVEN_SCENES_ROOT:-/data/mmc_syang/dataset/7scenes}"
NRGBD_ROOT="${NRGBD_ROOT:-/data/mmc_syang/dataset/NRGBD}"

case "$GPU" in
  2) METHOD="baseline"; PYTHON_BIN="$FAST_PYTHON"; METHOD_ARGS=(--omega-accelerator none) ;;
  3) METHOD="ours_adaptive_multiframe_fastvggt"; PYTHON_BIN="$FAST_PYTHON"; METHOD_ARGS=(
    --omega-accelerator none
    --enable-token-merging
    --token-merging-method frame_persistent_adaptive_spatial
    --token-merging-ratio 0.9
    --token-merging-layer-ratios '1-10:0.9,11-18:0.0,19-24:0.9'
    --token-merging-start 0
    --token-merging-frame-restore-layer 24
    --token-merging-frame-alpha 0.1
    --token-merging-frame-segment-threshold 0.9
    --token-merging-frame-merge-threshold 0.1
    --token-merging-frame-max-window 20
    --token-merging-frame-pool-stride 2
    --token-merging-frame-multi-max-group-size 4
    --token-merging-frame-multi-pair-threshold 0.986
    --token-merging-frame-multi-span-threshold 0.948
    --token-merging-frame-group-strategy local
  ) ;;
  4) METHOD="fastvggt_r090"; PYTHON_BIN="$FAST_PYTHON"; METHOD_ARGS=(
    --omega-accelerator none
    --enable-token-merging
    --token-merging-method spatial
    --token-merging-ratio 0.9
    --token-merging-start 0
  ) ;;
  5) METHOD="sparse_vggt"; PYTHON_BIN="$SPARSE_PYTHON"; METHOD_ARGS=(
    --omega-accelerator sparse_vggt
    --sparse-vggt-sparse-ratio 0.5
    --sparse-vggt-pool-mode avg
  ) ;;
  6) METHOD="da_vggt"; PYTHON_BIN="$FAST_PYTHON"; METHOD_ARGS=(
    --omega-accelerator da_vggt
    --da-vggt-max-frames 64
    --da-vggt-sampling-method fl_maxmin
    --da-vggt-n-anchors 1
    --da-vggt-lambda-div 0.0
  ) ;;
  7) METHOD="da_vggt_fastvggt_r090"; PYTHON_BIN="$FAST_PYTHON"; METHOD_ARGS=(
    --omega-accelerator da_vggt
    --da-vggt-max-frames 64
    --da-vggt-sampling-method fl_maxmin
    --da-vggt-n-anchors 1
    --da-vggt-lambda-div 0.0
    --enable-token-merging
    --token-merging-method spatial
    --token-merging-ratio 0.9
    --token-merging-start 0
  ) ;;
  *) echo "Expected GPU 2, 3, 4, 5, 6, or 7; got $GPU" >&2; exit 2 ;;
esac

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/$METHOD"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

run_case() {
  local label="$1"
  local dataset="$2"
  local dataset_root="$3"
  shift 3
  local log="$RUN_ROOT/logs/${METHOD}_${label}_gpu${GPU}.log"

  echo "[$(date '+%F %T')] start method=${METHOD} dataset=${label} gpu=${GPU}" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT:$SPARSE_REPO/src:$SPARSE_REPO/external/vggt:$SPARSE_REPO/external/SpargeAttn:$SPARSE_REPO" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" "$ROOT/inference/infer.py" \
    --dataset "$dataset" \
    --dataset-root "$dataset_root" \
    --output-dir "$RUN_ROOT/$METHOD" \
    --max-frames-per-seq 300 \
    --window-size 0 \
    --checkpoint "$CHECKPOINT" \
    --overwrite \
    --eval \
    --eval-align scale_shift \
    --pose-eval-frames 0 \
    --pose-eval-seed 0 \
    "${METHOD_ARGS[@]}" \
    "$@" \
    >> "$log" 2>&1
  echo "[$(date '+%F %T')] complete method=${METHOD} dataset=${label} gpu=${GPU}" | tee -a "$log"
}

run_case tum300 tum_dynamic "$TUM_ROOT"
run_case 7scenes_test300 7scenes "$SEVEN_SCENES_ROOT" --seven-scenes-split test
run_case nrgbd300 nrgbd "$NRGBD_ROOT"
