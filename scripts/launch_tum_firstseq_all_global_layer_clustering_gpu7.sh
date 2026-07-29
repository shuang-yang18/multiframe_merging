#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/new_results/1/tum_firstseq_uniform10_all_global_layers_pca128_k03}"
SESSION="${SESSION:-vggt_tum_all_global_cluster_gpu7}"

mkdir -p "$RUN_ROOT/logs"
screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -lc "cd '$ROOT' && GPU=7 RUN_ROOT='$RUN_ROOT' bash scripts/run_tum_firstseq_all_global_layer_clustering.sh"
printf 'screen session: %s\nresults: %s\n' "$SESSION" "$RUN_ROOT"
