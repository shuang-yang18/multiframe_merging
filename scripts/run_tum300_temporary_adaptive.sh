#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_tum300_temporary_adaptive.sh <gpu>}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/01/temporary_adaptive_p0988_s0955}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}"

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

echo "[$(date '+%F %T')] start TUM300 temporary adaptive fusion on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" inference/infer.py \
  --dataset tum_dynamic \
  --dataset-root "${TUM_ROOT:-$ROOT/datasets/TUM-Dynamics}" \
  --max-frames-per-seq 300 \
  --window-size 0 \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RUN_ROOT" \
  --overwrite \
  --eval \
  --eval-align scale_shift \
  --pose-eval-frames 0 \
  --pose-eval-seed 0 \
  --omega-accelerator none \
  --enable-token-merging \
  --token-merging-method frame_temporary_adaptive_spatial \
  --token-merging-layer-ratios '1-10:0.9,11-18:0.0,19-24:0.9' \
  --token-merging-frame-alpha 0.1 \
  --token-merging-frame-segment-threshold 0.9 \
  --token-merging-frame-max-window 20 \
  --token-merging-frame-pool-stride 2 \
  --token-merging-frame-multi-max-group-size 4 \
  --token-merging-frame-multi-pair-threshold 0.988 \
  --token-merging-frame-multi-span-threshold 0.955 \
  --token-merging-frame-group-strategy local
echo "[$(date '+%F %T')] complete TUM300 temporary adaptive fusion"
