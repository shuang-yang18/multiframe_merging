#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
GPU="${GPU:-5}"
SCHEDULE="${SCHEDULE:-1-10:0.9,11-18:0.3,19-24:0.9}"
CHECKPOINT="${CHECKPOINT:-checkpoints/vggt_omega_1b_512.pt}"

cd "$ROOT"
mkdir -p outputs/logs
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

run_eval() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] start ${name} gpu=${GPU} schedule=${SCHEDULE}"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT" \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" inference/infer.py \
    --dataset tum_dynamic \
    --output-dir "outputs/${name}" \
    --max-frames-per-seq 300 \
    --window-size 0 \
    --checkpoint "$CHECKPOINT" \
    --overwrite \
    --eval \
    --enable-token-merging \
    --token-merging-ratio 0.9 \
    --token-merging-layer-ratios "$SCHEDULE" \
    --token-merging-start 0 \
    "$@"
  echo "[$(date '+%F %T')] done ${name}"
}

run_eval "tum300_spatial_layer_r090_030_090" \
  --token-merging-method spatial

run_eval "tum300_frame_persistent_spatial_k0_restore24_a010_s090_m010_w100_layer_r090_030_090" \
  --token-merging-method frame_persistent_spatial \
  --token-merging-frame-pool-stride 2 \
  --token-merging-frame-segment-threshold 0.9 \
  --token-merging-frame-merge-threshold 0.1 \
  --token-merging-frame-alpha 0.1 \
  --token-merging-frame-max-window 100 \
  --token-merging-frame-restore-layer 24
