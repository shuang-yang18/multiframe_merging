#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${1:?usage: run_tum300_frame_anchor_adaptive.sh <gpu> <plain|fastvggt>}"
VARIANT="${2:?usage: run_tum300_frame_anchor_adaptive.sh <gpu> <plain|fastvggt>}"
TUM_ROOT="${TUM_ROOT:-$ROOT/../dataset/TUM-Dynamics}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/frame_anchor_adaptive_tum300_20260723}"
ADAPTIVE_BOUNDARY_Z="${ADAPTIVE_BOUNDARY_Z:-2.5}"
ADAPTIVE_MEDOID_Z="${ADAPTIVE_MEDOID_Z:-1.5}"
PATCH_FUSION_QUANTILE="${PATCH_FUSION_QUANTILE:-0.75}"

case "$VARIANT" in
  plain)
    METHOD="frame_anchor_adaptive"
    DEFAULT_OUTPUT_NAME="tum300_frame_anchor_adaptive_medoid_patchfusion_restore24"
    ;;
  fastvggt)
    METHOD="frame_anchor_adaptive_spatial"
    DEFAULT_OUTPUT_NAME="tum300_frame_anchor_adaptive_medoid_patchfusion_fastvggt_r090_restore24"
    ;;
  *)
    echo "Unknown variant: $VARIANT (expected plain or fastvggt)" >&2
    exit 2
    ;;
esac
OUTPUT_NAME="${OUTPUT_NAME:-$DEFAULT_OUTPUT_NAME}"

mkdir -p "$RUN_ROOT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
HF_HOME="$ROOT/.cache/huggingface" \
TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/hub" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" inference/infer.py \
  --device cuda \
  --dataset tum_dynamic \
  --dataset-root "$TUM_ROOT" \
  --max-frames-per-seq 300 \
  --frame-sample-mode first \
  --window-size 0 \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir "$RUN_ROOT/$OUTPUT_NAME" \
  --overwrite \
  --eval \
  --eval-align scale_shift \
  --pose-eval-frames 0 \
  --pose-eval-seed 0 \
  --enable-token-merging \
  --token-merging-method "$METHOD" \
  --token-merging-start 0 \
  --token-merging-ratio 0.9 \
  --token-merging-frame-restore-layer 24 \
  --token-merging-frame-pool-stride 2 \
  --token-merging-frame-adaptive-boundary-z "$ADAPTIVE_BOUNDARY_Z" \
  --token-merging-frame-adaptive-medoid-z "$ADAPTIVE_MEDOID_Z" \
  --token-merging-frame-patch-fusion-quantile "$PATCH_FUSION_QUANTILE"
