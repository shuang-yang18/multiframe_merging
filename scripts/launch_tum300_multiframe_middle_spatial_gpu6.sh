#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/tum300_multiframe_k0_max4_p0986_s0948_middle_r090}"
SESSION="${SESSION:-vggt_tum300_multiframe_middle_gpu6}"

mkdir -p "$RUN_ROOT"
screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -lc "cd '$ROOT' && GPU=6 RUN_ROOT='$RUN_ROOT' bash scripts/run_tum300_multiframe_middle_spatial.sh"
printf 'screen session: %s\nresults: %s\n' "$SESSION" "$RUN_ROOT"
