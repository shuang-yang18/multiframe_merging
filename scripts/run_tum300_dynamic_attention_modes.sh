#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: run_tum300_dynamic_attention_modes.sh <gpu> <all|middle|late|middle_late>}"
MODE="${2:?usage: run_tum300_dynamic_attention_modes.sh <gpu> <all|middle|late|middle_late>}"
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
TUM_ROOT="${TUM_ROOT:-$ROOT/../dataset/TUM-Dynamics}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/tum300_dynamic_attention_modes_20260724}"

case "$MODE" in
  all|middle|late|middle_late) ;;
  *) echo "Unsupported dynamic attention mode: $MODE" >&2; exit 2 ;;
esac

mkdir -p "$RUN_ROOT/logs"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
PYTHONNOUSERSITE=1 \
HF_HOME="$ROOT/.cache/huggingface" \
TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" \
MPLCONFIGDIR="$ROOT/.cache/matplotlib" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" "$ROOT/inference/infer.py" \
  --dataset tum_dynamic \
  --dataset-root "$TUM_ROOT" \
  --output-dir "$RUN_ROOT/tum300_dynamic_attention_${MODE}" \
  --max-frames-per-seq 300 \
  --window-size 10 \
  --checkpoint "$ROOT/checkpoints/vggt_omega_1b_512.pt" \
  --dynamic-attention-mask "$MODE" \
  --eval \
  --eval-align scale_shift \
  --pose-eval-frames 0 \
  --pose-eval-seed 0 \
  --overwrite
