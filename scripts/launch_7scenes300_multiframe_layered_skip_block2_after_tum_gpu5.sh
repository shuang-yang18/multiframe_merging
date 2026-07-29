#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/2/7scenes_test300_multiframe_p0986_s0948_layered_r090_skip_block2}"
GPU="${1:-${GPU:-5}}"
WAIT_SESSION="${WAIT_SESSION:-vggt_tum300_multiframe_layered_skip2_gpu5}"
SESSION="${SESSION:-vggt_7scenes_multiframe_layered_skip2_gpu${GPU}}"

mkdir -p "$RUN_ROOT"
screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -lc "
  while screen -ls 2>/dev/null | grep -q '\\.${WAIT_SESSION}[[:space:]]'; do sleep 60; done
  cd '$ROOT'
  GPU='$GPU' RUN_ROOT='$RUN_ROOT' bash scripts/run_7scenes300_multiframe_layered_skip_block2.sh
"
printf 'screen session: %s (waiting for %s)\nresults: %s\n' "$SESSION" "$WAIT_SESSION" "$RUN_ROOT"
