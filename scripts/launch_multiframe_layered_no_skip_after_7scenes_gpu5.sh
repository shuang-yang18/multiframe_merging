#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/multiframe_p0986_s0948_layered_r090_no_skip_tum300_7scenes_test300}"
GPU="${1:-${GPU:-5}}"
WAIT_SESSION="${WAIT_SESSION:-vggt_7scenes_multiframe_layered_skip2_gpu5}"
SESSION="${SESSION:-vggt_multiframe_layered_no_skip_tum_7scenes_gpu${GPU}}"

mkdir -p "$RUN_ROOT"
screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -lc "
  while screen -ls 2>/dev/null | grep -q '\\.${WAIT_SESSION}[[:space:]]'; do sleep 60; done
  cd '$ROOT'
  GPU='$GPU' RUN_ROOT='$RUN_ROOT' bash scripts/run_multiframe_layered_no_skip_tum_7scenes300.sh
"
printf 'screen session: %s (waiting for %s)\nresults: %s\n' "$SESSION" "$WAIT_SESSION" "$RUN_ROOT"
