#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_bonn_test_random10_baseline.sh <gpu>}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/bonn_test_random10_baseline_vggt_official_v2_20260722}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"

mkdir -p "$RUN_ROOT"
cd "$ROOT"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" inference/infer.py \
  --dataset bonn \
  --dataset-root datasets/Bonn/rgbd_bonn_dataset \
  --bonn-rgb-dir rgb \
  --bonn-depth-dir depth \
  --bonn-association-max-diff 0.02 \
  --max-frames-per-seq 10 \
  --frame-sample-mode random \
  --frame-sample-seed 0 \
  --frame-sample-random-order \
  --window-size 0 \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir "$RUN_ROOT" \
  --eval \
  --pose-eval-frames 0 \
  --overwrite \
  --omega-accelerator none

PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
"$PYTHON_BIN" inference/eval_vggt_official_pose_auc.py \
  --dataset-root datasets/Bonn/rgbd_bonn_dataset \
  --pred-dir "$RUN_ROOT" \
  --dataset bonn
