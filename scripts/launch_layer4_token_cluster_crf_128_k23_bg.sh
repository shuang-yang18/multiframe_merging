#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/layer4_dynamic_segmentation_crf_tum_7scenes_test300_pca128_k02_k03}"
SESSION="${SESSION:-vggt_layer4_cluster_crf_128_k23}"

mkdir -p "$RUN_ROOT/logs"
screen -dmS "$SESSION" bash -lc \
  "cd '$ROOT' && RUN_ROOT='$RUN_ROOT' PCA_DIMS=128 CLUSTER_COUNTS=2,3 LABEL_SMOOTHING=crf CRF_SPATIAL_WEIGHT=0.9 CRF_TEMPORAL_WEIGHT=0.08 CRF_COLOR_SIGMA=0.18 CRF_UNARY_TEMPERATURE=1.0 CRF_ITERATIONS=10 bash scripts/run_layer4_token_cluster_grid.sh"
printf 'screen session: %s\nlogs: %s/logs\n' "$SESSION" "$RUN_ROOT"
