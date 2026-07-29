#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/tum300_baseline_all_global_attention}"
GPU="${1:-${GPU:-5}}"
SESSION="${SESSION:-vggt_tum300_all_global_gpu${GPU}}"

mkdir -p "$RUN_ROOT"
screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -lc "cd '$ROOT' && GPU='$GPU' RUN_ROOT='$RUN_ROOT' bash scripts/run_tum300_all_global_attention.sh"
printf 'screen session: %s\nresults: %s\n' "$SESSION" "$RUN_ROOT"
