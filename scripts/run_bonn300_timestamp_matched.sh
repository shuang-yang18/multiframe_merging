#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_bonn300_timestamp_matched.sh <gpu> <baseline|fastvggt|sparse_vggt|da_vggt>}"
METHOD="${2:?usage: run_bonn300_timestamp_matched.sh <gpu> <baseline|fastvggt|sparse_vggt|da_vggt>}"
RUN_ROOT="${RUN_ROOT:-$ROOT/auc_eval_results/bonn300_timestamp_matched_20260722}"
BONN_ROOT="${BONN_ROOT:-$ROOT/datasets/Bonn/rgbd_bonn_dataset}"
FAST_PYTHON="${FAST_PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
SPARSE_REPO="${SPARSE_REPO:-$ROOT/../sparse-vggt}"
SPARSE_PYTHON="${SPARSE_PYTHON:-$SPARSE_REPO/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}"

mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

OUTPUT_NAME=""
PYTHON_BIN="$FAST_PYTHON"
METHOD_ARGS=()
case "$METHOD" in
  baseline)
    OUTPUT_NAME="bonn300_baseline"
    METHOD_ARGS=(--omega-accelerator none)
    ;;
  fastvggt)
    OUTPUT_NAME="bonn300_fastvggt_r090"
    METHOD_ARGS=(
      --omega-accelerator none
      --enable-token-merging
      --token-merging-method spatial
      --token-merging-ratio 0.9
      --token-merging-start 0
    )
    ;;
  sparse_vggt)
    OUTPUT_NAME="bonn300_sparse_vggt"
    PYTHON_BIN="$SPARSE_PYTHON"
    METHOD_ARGS=(
      --omega-accelerator sparse_vggt
      --sparse-vggt-sparse-ratio 0.5
      --sparse-vggt-pool-mode avg
    )
    ;;
  da_vggt)
    OUTPUT_NAME="bonn300_da_vggt"
    METHOD_ARGS=(
      --omega-accelerator da_vggt
      --da-vggt-max-frames 64
      --da-vggt-sampling-method fl_maxmin
      --da-vggt-n-anchors 1
      --da-vggt-lambda-div 0.0
    )
    ;;
  *)
    echo "Unknown method: $METHOD" >&2
    exit 2
    ;;
esac

echo "[$(date '+%F %T')] start method=$METHOD gpu=$GPU output=$RUN_ROOT/$OUTPUT_NAME"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT:$SPARSE_REPO/src:$SPARSE_REPO/external/vggt:$SPARSE_REPO/external/SpargeAttn:$SPARSE_REPO" \
PYTHONNOUSERSITE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" inference/infer.py \
  --dataset bonn \
  --dataset-root "$BONN_ROOT" \
  --bonn-rgb-dir rgb \
  --bonn-depth-dir depth \
  --bonn-association-max-diff 0.02 \
  --max-frames-per-seq 300 \
  --window-size 0 \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RUN_ROOT/$OUTPUT_NAME" \
  --eval \
  --pose-eval-frames 0 \
  --pose-eval-seed 0 \
  --overwrite \
  "${METHOD_ARGS[@]}"
echo "[$(date '+%F %T')] done method=$METHOD gpu=$GPU output=$RUN_ROOT/$OUTPUT_NAME"
