#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SEARCH_DIR="${SEARCH_DIR:-$ROOT/new_results/2/tum300_layered_pair_span_r_search_p0986_s0948}"
SESSION="${SESSION:-vggt_tum300_layered_pair_span_r_gpu6}"

mkdir -p "$SEARCH_DIR"
screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -lc "cd '$ROOT' && \
  PYTHON=/data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  GPUS=6 SEARCH_DIR='$SEARCH_DIR' RESUME_SEARCH=0 TIME_BUDGET_HOURS=0 \
  PAIR_CENTER=0.986 SPAN_CENTER=0.948 R_CENTER=0.9 \
  /data/mmc_syang/miniconda3/envs/fastvggt/bin/python scripts/search_pair_span_r_tum.py"
printf 'screen session: %s\nresults: %s\n' "$SESSION" "$SEARCH_DIR"
