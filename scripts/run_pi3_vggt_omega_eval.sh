#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/flow3r/bin/python}"
GPU="${1:-0}"
OUTPUT_DIR="${2:?usage: $0 <gpu> <output-dir> [extra run_pi3_vggt_omega_eval.py arguments]}"
shift 2

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT" \
HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}" \
"$PYTHON_BIN" scripts/run_pi3_vggt_omega_eval.py \
  --dataset all \
  --output-dir "$OUTPUT_DIR" \
  --max-frames-per-seq 300 \
  --pretrained checkpoints/Pi3 \
  "$@"
