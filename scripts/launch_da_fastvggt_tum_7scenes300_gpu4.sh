#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/da_fastvggt_r090_tum300_7scenes_test300}"
GPU="${1:-${GPU:-4}}"
SESSION="${SESSION:-vggt_da_fastvggt_tum_7scenes_gpu${GPU}}"

mkdir -p "$RUN_ROOT"
screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -lc "cd '$ROOT' && GPU='$GPU' RUN_ROOT='$RUN_ROOT' bash scripts/run_da_fastvggt_tum_7scenes300.sh"
printf 'screen session: %s\nresults: %s\n' "$SESSION" "$RUN_ROOT"
