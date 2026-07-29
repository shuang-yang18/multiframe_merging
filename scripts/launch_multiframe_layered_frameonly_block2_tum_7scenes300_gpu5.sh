#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/multiframe_p0986_s0948_layered_r090_frameonly_block2_tum300_7scenes_test300}"
GPU="${1:-${GPU:-5}}"
SESSION="${SESSION:-vggt_multiframe_layered_frameonly2_tum_7scenes_gpu${GPU}}"

mkdir -p "$RUN_ROOT"
screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -lc "cd '$ROOT' && GPU='$GPU' RUN_ROOT='$RUN_ROOT' bash scripts/run_multiframe_layered_frameonly_block2_tum_7scenes300.sh"
printf 'screen session: %s\nresults: %s\n' "$SESSION" "$RUN_ROOT"
