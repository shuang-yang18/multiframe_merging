#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/layer4_dynamic_segmentation_potts_tum_7scenes_test300_pca128_k02_k03}"
SESSION="${SESSION:-vggt_layer4_cluster_potts_128_k23}"

mkdir -p "$RUN_ROOT/logs"
screen -dmS "$SESSION" bash -lc \
  "cd '$ROOT' && RUN_ROOT='$RUN_ROOT' PCA_DIMS=128 CLUSTER_COUNTS=2,3 LABEL_SMOOTHING=potts POTTS_SPATIAL_WEIGHT=0.12 POTTS_TEMPORAL_WEIGHT=0.06 POTTS_ITERATIONS=5 bash scripts/run_layer4_token_cluster_grid.sh"
printf 'screen session: %s\nlogs: %s/logs\n' "$SESSION" "$RUN_ROOT"
