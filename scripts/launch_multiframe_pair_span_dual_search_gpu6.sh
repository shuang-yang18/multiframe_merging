#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/multiframe_pair_span_dual_search_fastall_r090_skip2}"
GPU="${1:-${GPU:-6}}"
SESSION="${SESSION:-vggt_multiframe_pair_span_dual_search_gpu${GPU}}"

mkdir -p "$RUN_ROOT"
screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -lc "cd '$ROOT' && /data/mmc_syang/miniconda3/envs/fastvggt/bin/python scripts/search_multiframe_pair_span_dual_dataset.py --gpu '$GPU' --output-root '$RUN_ROOT' > '$RUN_ROOT/search.log' 2>&1"
printf 'screen session: %s\nresults: %s\n' "$SESSION" "$RUN_ROOT"
