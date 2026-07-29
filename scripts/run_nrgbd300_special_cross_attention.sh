#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:-5}"
ALPHA="${2:-0.1}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/nrgbd300_special_cross_attention_20260723}"
NRGBD_ROOT="${NRGBD_ROOT:-/data/mmc_syang/dataset/NRGBD}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
OUTPUT_NAME="nrgbd300_ours_special_cross_a${ALPHA/./}"

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

echo "[$(date '+%F %T')] start NRGBD300 special-cross-attention alpha=${ALPHA} on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" inference/infer.py \
  --dataset nrgbd \
  --dataset-root "$NRGBD_ROOT" \
  --max-frames-per-seq 300 \
  --window-size 0 \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RUN_ROOT/$OUTPUT_NAME" \
  --eval \
  --pose-eval-frames 0 \
  --pose-eval-seed 0 \
  --overwrite \
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
  --token-merging-frame-group-strategy local \
  --token-merging-frame-special-cross-attention \
  --token-merging-frame-special-cross-attention-alpha "$ALPHA"
echo "[$(date '+%F %T')] complete NRGBD300 special-cross-attention alpha=${ALPHA} on GPU ${GPU}"
