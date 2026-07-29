#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:-7}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/segment_patch_bank_v1_tum300}"
TUM_ROOT="${TUM_ROOT:-$ROOT/../dataset/TUM-Dynamics}"
PYTHON="${PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"

mkdir -p "$RUN_ROOT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
HF_HOME="$ROOT/.cache/huggingface" \
TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" inference/infer.py \
  --device cuda \
  --dataset tum_dynamic \
  --dataset-root "$TUM_ROOT" \
  --max-frames-per-seq 300 \
  --frame-sample-mode first \
  --window-size 0 \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir "$RUN_ROOT" \
  --eval \
  --eval-align scale_shift \
  --pose-eval-frames 0 \
  --pose-eval-seed 0 \
  --omega-accelerator none \
  --enable-token-merging \
  --token-merging-method segment_patch_bank \
  --token-merging-start 0 \
  --token-merging-segment-bank-pair-threshold 0.986 \
  --token-merging-segment-bank-span-threshold 0.948 \
  --token-merging-segment-bank-max-group-size 4
