#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXP_DIR="${EXP_DIR:-$ROOT/new_results/2/tum300_single_global_layer_spatial_r090}"
SESSION="${SESSION:-vggt_tum300_single_global_spatial_gpu7}"

mkdir -p "$EXP_DIR/logs"
screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -lc "cd '$ROOT' && GPU=7 EXP_DIR='$EXP_DIR' /data/mmc_syang/miniconda3/envs/fastvggt/bin/python scripts/run_tum300_single_global_layer_spatial.py"
printf 'screen session: %s\nresults: %s\n' "$SESSION" "$EXP_DIR"
