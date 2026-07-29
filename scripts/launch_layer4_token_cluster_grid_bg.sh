#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/layer4_dynamic_segmentation_tum_7scenes_test300_uniform10}"
SESSION="${SESSION:-vggt_layer4_cluster_grid}"

mkdir -p "$RUN_ROOT/logs"
screen -dmS "$SESSION" bash -lc "cd '$ROOT' && bash scripts/run_layer4_token_cluster_grid.sh"
printf 'screen session: %s\nlogs: %s/logs\n' "$SESSION" "$RUN_ROOT"
