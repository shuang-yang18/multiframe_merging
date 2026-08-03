#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_legacy_adaptive_policy_eval.sh <gpu> <tum|7scenes|nrgbd>}"
DATASET_KEY="${2:?usage: run_legacy_adaptive_policy_eval.sh <gpu> <tum|7scenes|nrgbd>}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/legacy_adaptive_policy_20260731}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}"

DATASET_ARGS=()
case "$DATASET_KEY" in
  tum)
    DATASET_ARGS=(--dataset tum_dynamic --dataset-root "${TUM_ROOT:-$ROOT/datasets/TUM-Dynamics}")
    OUTPUT_NAME="tum300_legacy_adaptive_policy"
    ;;
  7scenes)
    DATASET_ARGS=(
      --dataset 7scenes
      --dataset-root "${SEVEN_SCENES_ROOT:-$ROOT/datasets/7scenes/test}"
      --seven-scenes-split test
    )
    OUTPUT_NAME="7scenes_test300_legacy_adaptive_policy"
    ;;
  nrgbd)
    DATASET_ARGS=(--dataset nrgbd --dataset-root "${NRGBD_ROOT:-/data/mmc_syang/dataset/NRGBD}")
    OUTPUT_NAME="nrgbd300_legacy_adaptive_policy"
    ;;
  *)
    echo "Expected dataset tum, 7scenes, or nrgbd; got $DATASET_KEY" >&2
    exit 2
    ;;
esac

cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

echo "[$(date '+%F %T')] start ${DATASET_KEY}300 legacy adaptive policy on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" inference/infer.py \
  "${DATASET_ARGS[@]}" \
  --max-frames-per-seq 300 \
  --window-size 0 \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RUN_ROOT/$OUTPUT_NAME" \
  --overwrite \
  --eval \
  --pose-eval-frames 0 \
  --pose-eval-seed 0 \
  --omega-accelerator none \
  --enable-token-merging \
  --token-merging-method frame_persistent_adaptive_spatial \
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
echo "[$(date '+%F %T')] complete ${DATASET_KEY}300 legacy adaptive policy on GPU ${GPU}"
